import streamlit as st
import ollama
import json
import re

# =============================
# CONFIGURATION
# =============================
MODEL = "qwen2.5"  # Change to your local model name if needed


# =============================
# HELPER FUNCTIONS
# =============================

def chatbot_response(user_message):
    """Generate chatbot response."""
    prompt = f"You are a fintech debt collection assistant. Respond empathetically and professionally.\n\nUser: {user_message}\nAssistant:"
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()


def evaluate_response(user_message, bot_response, previous_user_message=None):
    """Evaluate bot response for compliance, tone, and factual accuracy."""
    eval_prompt = f"""
You are an evaluation agent checking chatbot messages in a fintech debt collection context.

TASK:
Rate the following chatbot response on:
1. **Compliance**: Legal and ethical accuracy around debt collection.
2. **Tone**: Empathetic, respectful, professional.
3. **Factuality**: The advice must be realistic and relevant to debt management.

INPUT:
Previous User Message: {previous_user_message if previous_user_message else "N/A"}
Current User Message: {user_message}
Chatbot Response: {bot_response}

OUTPUT FORMAT (strict JSON):
{{
    "compliance_ok": true/false,
    "tone_ok": true/false,
    "factuality_ok": true/false,
    "needs_correction": true/false,
    "feedback": "Detailed explanation of issues or validation."
}}
"""
    eval_result = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": eval_prompt}],
        options={"temperature": 0.1}
    )

    raw_text = eval_result["message"]["content"].strip()
    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if json_match:
        raw_text = json_match.group(0)

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = {
            "compliance_ok": False,
            "tone_ok": False,
            "factuality_ok": False,
            "needs_correction": True,
            "feedback": "Evaluator returned invalid JSON, assuming unsafe response."
        }

    return parsed


def correction_agent(user_message, bot_response, evaluation, previous_user_message=None):
    """Correct chatbot response if needed."""
    correction_prompt = f"""
You are a correction agent for a fintech debt collection chatbot.
Fix compliance, tone, or factual issues in the assistant’s message.

CONTEXT:
Previous User Message: {previous_user_message if previous_user_message else "N/A"}
Current User Message: {user_message}
Original Chatbot Response: {bot_response}
Evaluator Feedback: {evaluation['feedback']}

TASK:
Rewrite the chatbot response to be compliant, empathetic, and factually correct.
Return only the corrected message.
"""
    correction = ollama.chat(model=MODEL, messages=[{"role": "user", "content": correction_prompt}])
    return correction["message"]["content"].strip()


def run_chat_pipeline(user_message, previous_user_message=None):
    """Run chatbot + evaluation + correction."""
    bot_response = chatbot_response(user_message)
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
        "status": status
    }


# =============================
# STREAMLIT CHAT APP
# =============================

st.set_page_config(page_title="💬 Fintech Debt Chatbot", layout="centered")
st.title("💬 Fintech Debt Chatbot Evaluator")
st.caption("Test the chatbot in real time — every response is automatically evaluated.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_user_message" not in st.session_state:
    st.session_state.last_user_message = None

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input field
if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run full pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_chat_pipeline(user_input, st.session_state.last_user_message)
            response = result["final_response"]
            st.markdown(response)

            eval_data = result["evaluation"]

            # Convert boolean results to scores
            def score(flag): return 100 if flag else 40
            compliance_score = score(eval_data["compliance_ok"])
            tone_score = score(eval_data["tone_ok"])
            factuality_score = score(eval_data["factuality_ok"])

            avg_score = round((compliance_score + tone_score + factuality_score) / 3, 1)

            st.markdown("### 📊 Evaluation Summary")
            st.progress(avg_score / 100)
            col1, col2, col3 = st.columns(3)
            col1.metric("Compliance", f"{compliance_score}/100")
            col2.metric("Tone", f"{tone_score}/100")
            col3.metric("Factuality", f"{factuality_score}/100")

            st.info(eval_data["feedback"])

    # Save assistant message + update memory
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.last_user_message = user_input
