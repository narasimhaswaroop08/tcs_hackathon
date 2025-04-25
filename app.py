import streamlit as st
from backend import generate_llm_response_with_history, get_chat_history, save_his

# Page setup
st.set_page_config(page_title="Insurance policy Chatbot", layout="centered")

# Gemini theme CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');

        html, body {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: #ffffff;
            font-family: 'Orbitron', sans-serif;
        }

        .stChatMessage {
            padding: 15px;
            border-radius: 15px;
            margin-bottom: 10px;
            box-shadow: 0 4px 15px rgba(0, 255, 255, 0.2);
        }

        .user-msg {
            background: rgba(0, 230, 255, 0.2);
            border: 1px solid #00e5ff;
            color: #00e5ff;
        }

        .assistant-msg {
            background: rgba(138, 43, 226, 0.2);
            border: 1px solid #8a2be2;
            color: #ba68c8;
        }

        .stTextInput>div>div>input {
            background-color: #1c1f26;
            color: white;
            border-radius: 10px;
        }

        .stTextInput>div>div {
            border: 1px solid #00e5ff;
        }

        h1 {
            color: #00e5ff;
            text-align: center;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Session state
if "chat_started" not in st.session_state:
    st.session_state.chat_started = True
    st.session_state.history = []

# Gemini-style title
st.markdown("<h1>🌌Insurance policy Chatbot</h1>", unsafe_allow_html=True)

# Show chat history
with st.container():
    for item in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(f"<div class='stChatMessage user-msg'>{item['user']}</div>", unsafe_allow_html=True)
        with st.chat_message("assistant"):
            st.markdown(f"<div class='stChatMessage assistant-msg'>{item['assistant']}</div>", unsafe_allow_html=True)

# Input field
user_input = st.chat_input("Ask your insurance-related question...")

if user_input:
    with st.chat_message("user"):
        st.markdown(f"<div class='stChatMessage user-msg'>{user_input}</div>", unsafe_allow_html=True)

    response = generate_llm_response_with_history(user_input)
    a = save_his()
    with st.chat_message("assistant"):
        st.markdown(f"<div class='stChatMessage assistant-msg'>{response}</div>", unsafe_allow_html=True)

    st.session_state.history.append({
        "user": user_input,
        "assistant": response
    })
    
