"""AI Assistant page (Mlinzi)."""

import streamlit as st

from app_lib.chat import build_system_prompt, get_groq_client, render_message
from app_lib.config import GROQ_MODEL, get_secret
from app_lib.state import get_state

state = get_state()

st.markdown(
    '<div class="section-header">Flood Guard AI Assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(
    "Hi. My name is Mlinzi, an AI Chatbot specifically designed to help you "
    "with any questions you might have regarding flooding in Kenya. "
    "Ask anything about flood risk, affected wards, route recommendations, "
    "or how to interpret the model results."
)

if not get_secret("GROQ_API_KEY"):
    st.warning(
        "GROQ_API_KEY is not configured, so the assistant is unavailable. "
        "Add it to `.streamlit/secrets.toml` (or the deployment's secrets) "
        "to enable Mlinzi."
    )
    st.stop()

# Input form at the top
with st.form(key="chat_form", clear_on_submit=True):
    input_col, btn_col = st.columns([11, 1])
    with input_col:
        user_input = st.text_input(
            "",
            placeholder="Ask about flood risk, routes, or the model...",
            label_visibility="collapsed",
        )
    with btn_col:
        submitted = st.form_submit_button("➤")

st.markdown(
    "<hr style='border:none;border-top:1px solid #1F4A32;margin:0.5rem 0 1rem 0;'>",
    unsafe_allow_html=True,
)

# Process new input
if submitted and user_input.strip():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})
    with st.spinner("Mlinzi is thinking..."):
        # Context is built here, only when a message is actually being
        # sent to the LLM, rather than on every rerun of this page.
        system_prompt = build_system_prompt(state)

        client = get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                *st.session_state.messages,
            ],
            max_tokens=1024,
            temperature=0.4,
        )
        reply = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": reply})
    # Cap history so a long conversation doesn't grow memory (or the
    # Groq context sent on every turn) without bound.
    st.session_state.messages = st.session_state.messages[-20:]

# Render message history below the input
for msg in st.session_state.get("messages", []):
    render_message(msg["role"], msg["content"])

if st.session_state.get("messages"):
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
