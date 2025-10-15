import streamlit as st
import ollama
import json
import re

# =============================
# CONFIGURATION
# =============================
MODEL = "qwen2.5"
client = ollama.Client(host='http://localhost:11434')  # ✅ Reuse a single client across all calls

# =============================
# SHARED CONTEXTS
# =============================

PERSONA = """
#Persona:
You are an empathetic and professional debt collection assistant. You understand the stress and challenges that come with debt, and you aim to provide clear, respectful, and helpful guidance to customers with your aim being to genuinely help them with a passion to do so.
"""

CUSTOMER_CONTEXT = """
Customer details:
- Name: Atyab Tosif
- Debt Amount: $5,000
- Due Date: 2023-12-31
- Last Payment Date: 2023-10-15
- Payment History: On-time payments for 10 months, missed last payment
"""

SOLUTIONS_CATALOGUE = """
Solutions Catalogue:
1. Payment Plan - Spread the debt over 6 to 12 months with no interest.
2. Settlement Offer - Pay a lump sum of 70 percent of the debt to clear it.
3. Financial Counseling - Free advice and budgeting help.
4. Hardship Program - Temporary payment deferral up to 3 months.
5. Automatic Payments - Set up automatic monthly payments.
6. Online Account Management - View balance, make payments, manage account.
7. Customer Support - 24/7 phone, email, or chat support.
"""

# =============================
# HELPER FUNCTIONS
# =============================

def stream_chat_response(messages):
    """Stream the chatbot response for faster perceived response."""
    full_reply = ""
    stream = client.chat(model=MODEL, messages=messages, stream=True)
    for chunk in stream:
        if "message" in chunk and "content" in chunk["message"]:
            content = chunk["message"]["content"]
            full_reply += content
            yield content  # Stream chunk to UI
    return full_reply


def chatbot_response(user_message):
    """Generate chatbot response using streaming."""
    system_prompt = f"""
{PERSONA}

#Job:
You work for a fintech company that offers debt collection services. 
Your role is to assist customers in managing their debt effectively and ethically.

#Guardrails:
1. Never be impolite, rude or dismissive.
2. Never offer financial advice that is not in the solutions catalogue.
3. Never share personal data or sensitive information.
4. Always prioritize the customer's best interests and well-being.
5. If unsure, ask for clarification or suggest they speak to a human agent.

#Environment:
You work in a fintech company aiming to improve debt collection experiences through empathy and professionalism.

{CUSTOMER_CONTEXT}

{SOLUTIONS_CATALOGUE}

#Goal:
Show that debt collection can be positive when handled with empathy, professionalism, and genuine intent to help.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # Stream response and accumulate
    response = ""
    for chunk in stream_chat_response(messages):
        response += chunk
        yield chunk  # ✅ Stream to UI in real time

    return response.strip()


def safe_json_parse(raw_text):
    """Safely extract and parse JSON from raw LLM output."""
    # Remove code block wrappers if present
    raw = raw_text.strip()
    if "```" in raw:
        raw = raw.split("```")[-2] if "```" in raw else raw
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "compliance_ok": False,
            "tone_ok": False,
            "factuality_ok": False,
            "needs_correction": True,
            "feedback": "Evaluator returned invalid JSON; assumed unsafe response."
        }


def evaluate_response(user_message, bot_response, previous_user_message=None):
    """Evaluate bot response for compliance, tone, and factual accuracy."""
    system_prompt = f"""
You are a human-in-the-loop evaluation agent reviewing a debt collection chatbot's responses.

#Persona:
Detail-oriented, compliance-focused, and empathetic reviewer ensuring tone and factuality.
{CUSTOMER_CONTEXT}
{SOLUTIONS_CATALOGUE}
"""

    user_prompt = f"""
Evaluate the chatbot response below on:
1. Compliance
2. Tone
3. Factuality

Previous User Message: {previous_user_message or "N/A"}
Current User Message: {user_message}
Chatbot Response: {bot_response}

#Output Format (strict JSON)
{{
    "compliance_ok": true/false,
    "tone_ok": true/false,
    "factuality_ok": true/false,
    "needs_correction": true/false,
    "feedback": "Detailed explanation of issues or validation."
}}
"""
    result = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.1}
    )

    return safe_json_parse(result["message"]["content"])


def correction_agent(user_message, bot_response, evaluation, previous_user_message=None):
    """Correct chatbot response if needed."""
    system_prompt = f"""
You are a correction agent responsible for improving a debt collection chatbot's responses.

#Persona:
Empathetic, compliant, professional.
{CUSTOMER_CONTEXT}
{SOLUTIONS_CATALOGUE}
"""

    user_prompt = f"""
Previous User Message: {previous_user_message or "N/A"}
Current User Message: {user_message}

Original Chatbot Response:
{bot_response}

Evaluator Feedback:
{evaluation.get("feedback", "No feedback provided.")}

#Correction Objective:
Fix all issues related to compliance, tone, and factuality.
Return only the corrected chatbot response.
"""

    result = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.3}
    )
    return result["message"]["content"].strip()


def run_chat_pipeline(user_message, previous_user_message=None):
    """Run chatbot + evaluation + correction."""
    # Stream response directly to UI
    bot_response = ""
    response_placeholder = st.empty()
    for chunk in chatbot_response(user_message):
        bot_response += chunk
        response_placeholder.markdown(bot_response)

    evaluation = evaluate_response(user_message, bot_response, previous_user_message)
    if evaluation.get("needs_correction", False):
        corrected_response = correction_agent(user_message, bot_response, evaluation, previous_user_message)
        final_response = corrected_response
        status = "Corrected"
    else:
        final_response = bot_response
        status = "Safe"

    return {
        "user_message": user_message,
        "previous_user_message": previous_user_message,
        "original_response": bot_response,
        "evaluation": evaluation,
        "final_response": final_response,
        "status": status,
    }

# =============================
# STREAMLIT CHAT APP
# =============================

st.set_page_config(page_title="💬 Fintech Debt Chatbot", layout="centered")
st.title("💬 Fintech Debt Chatbot Evaluator")
st.caption("Test the chatbot in real time — every response is automatically evaluated.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_user_message" not in st.session_state:
    st.session_state.last_user_message = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_chat_pipeline(user_input, st.session_state.last_user_message)
            response = result["final_response"]

            eval_data = result["evaluation"]
            compliance_score = 100 if eval_data["compliance_ok"] else 40
            tone_score = 100 if eval_data["tone_ok"] else 40
            factuality_score = 100 if eval_data["factuality_ok"] else 40
            avg_score = round((compliance_score + tone_score + factuality_score) / 3, 1)

            st.markdown("### 📊 Evaluation Summary")
            st.progress(avg_score / 100)
            col1, col2, col3 = st.columns(3)
            col1.metric("Compliance", f"{compliance_score}/100")
            col2.metric("Tone", f"{tone_score}/100")
            col3.metric("Factuality", f"{factuality_score}/100")
            st.info(eval_data["feedback"])

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.last_user_message = user_input