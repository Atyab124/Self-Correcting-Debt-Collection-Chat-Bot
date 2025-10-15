import streamlit as st
import ollama
import json
import re

# =============================
# CONFIGURATION
# =============================
MODEL = "qwen2.5"
MAX_CONTEXT_MESSAGES = 6  # ✅ Limit number of prior messages to reduce VRAM usage
client = ollama.Client(host='http://localhost:11434')

# =============================
# SHARED CONTEXTS
# =============================

PERSONA = """
# Persona
Your name is **Sia**, an empathetic and professional debt collection assistant working for **Company X** — 
a fintech firm that specializes in ethical, customer-first debt management.

You represent Company X when speaking to customers, 
and your goal is to turn debt resolution into a positive, supportive experience. 
You reach out to customers regarding overdue payments, introducing yourself politely, 
explaining the reason for contact, and offering solutions calmly and clearly.
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

def build_message_history(user_message):
    """Build a truncated, context-aware chat history to pass to the LLM."""
    history = [
        {"role": "system", "content": f"""
{PERSONA}

# Job
You work for Company X, a fintech firm offering ethical debt collection services. 
You are reaching out to the debtor proactively on behalf of Company X.

# Guardrails
1. Never be impolite, rude, or dismissive.
2. Never offer financial advice outside the Solutions Catalogue.
3. Never share personal data or sensitive information.
4. Always prioritize the customer's well-being.
5. If unsure, ask for clarification or suggest they speak to a human agent.

{CUSTOMER_CONTEXT}
{SOLUTIONS_CATALOGUE}

# Goal
Reach out to the debtor empathetically, acknowledge their situation, 
and guide them toward a suitable solution from Company X.
"""}
    ]

    # ✅ Include only the most recent N messages for VRAM efficiency
    prior_msgs = st.session_state.messages[-MAX_CONTEXT_MESSAGES:]
    for msg in prior_msgs:
        if msg["role"] in ("user", "assistant"):
            history.append({"role": msg["role"], "content": msg["content"]})

    # Append the latest user message
    history.append({"role": "user", "content": user_message})
    return history


def stream_chat_response(messages):
    """Stream the chatbot response for faster perceived response."""
    full_reply = ""
    stream = client.chat(model=MODEL, messages=messages, stream=True)
    for chunk in stream:
        if "message" in chunk and "content" in chunk["message"]:
            content = chunk["message"]["content"]
            full_reply += content
            yield content
    return full_reply


def chatbot_response(user_message):
    """Generate chatbot response using streaming and context."""
    messages = build_message_history(user_message)
    response = ""
    for chunk in stream_chat_response(messages):
        response += chunk
        yield chunk
    return response.strip()


def safe_json_parse(raw_text):
    """Safely extract and parse JSON from raw LLM output."""
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

# Persona
You are compliance-focused, empathetic, and detail-oriented.
You work at Company X, ensuring its debt collection chatbot (Sia) is safe and compliant.

{CUSTOMER_CONTEXT}
{SOLUTIONS_CATALOGUE}
"""

    user_prompt = f"""
Evaluate the chatbot response below on:
1. Compliance (legal and ethical)
2. Tone (empathetic and respectful)
3. Factuality (accurate, consistent with provided information)

Previous User Message: {previous_user_message or "N/A"}
Current User Message: {user_message}
Chatbot Response: {bot_response}

# Output Format (strict JSON)
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
You are a correction agent responsible for improving Company X’s debt collection chatbot responses.

# Persona
Empathetic, compliant, professional.
Work for Company X to ensure tone, factuality, and ethics are perfect.

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

# Correction Objective
Fix all issues related to compliance, tone, and factuality.
Return only the corrected chatbot response (no metadata).
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
    bot_response = ""
    response_placeholder = st.empty()

    for chunk in chatbot_response(user_message):
        bot_response += chunk
        response_placeholder.markdown(bot_response)

    evaluation = evaluate_response(user_message, bot_response, previous_user_message)

    corrected_response = None
    status = "Safe"
    if evaluation.get("needs_correction", False):
        corrected_response = correction_agent(user_message, bot_response, evaluation, previous_user_message)
        status = "Corrected"

    final_response = corrected_response or bot_response

    return {
        "user_message": user_message,
        "previous_user_message": previous_user_message,
        "original_response": bot_response,
        "evaluation": evaluation,
        "corrected_response": corrected_response,
        "final_response": final_response,
        "status": status,
    }

# =============================
# STREAMLIT CHAT APP
# =============================

st.set_page_config(page_title="💬 Company X Debt Chatbot", layout="centered")
st.title("💬 Company X Debt Chatbot Evaluator")
st.caption("Sia — your empathetic debt collection assistant from Company X. Every response is evaluated and corrected if needed.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_user_message" not in st.session_state:
    st.session_state.last_user_message = None

# ✅ Auto-initiate conversation (Sia greets first)
if len(st.session_state.messages) == 0:
    with st.spinner("Sia is preparing her first message..."):
        greeting = "Hello Atyab, this is Sia from Company X. I wanted to check in with you regarding your account. I’m here to help make the repayment process as easy as possible. How have things been going lately?"
        result = run_chat_pipeline(greeting)
        st.session_state.messages.append({"role": "assistant", "content": result["final_response"]})
        st.session_state.last_user_message = None
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Sia is composing her message..."):
            result = run_chat_pipeline(user_input, st.session_state.last_user_message)

            st.markdown("### 🤖 Original Response")
            st.write(result["original_response"])

            eval_data = result["evaluation"]
            st.markdown("### 📊 Evaluation Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Compliance", "✅" if eval_data["compliance_ok"] else "❌")
            col2.metric("Tone", "✅" if eval_data["tone_ok"] else "❌")
            col3.metric("Factuality", "✅" if eval_data["factuality_ok"] else "❌")
            st.info(eval_data["feedback"])

            if result["status"] == "Corrected":
                st.markdown("---")
                st.markdown("### ✨ Corrected Response")
                st.write(result["corrected_response"])

    st.session_state.messages.append({"role": "assistant", "content": result["final_response"]})
    st.session_state.last_user_message = user_input
