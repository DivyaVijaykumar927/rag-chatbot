import streamlit as st
import json
import os
from src.rag_engine import retrieve_and_answer
from src.persona_extractor import load_persona

# Page setup
st.set_page_config(page_title="Conversation RAG Chatbot", page_icon="🤖", layout="wide")
st.title(" Conversation Analysis Chatbot")
st.markdown("Ask me anything about the user based on their conversation history!")

# Load persona
persona = load_persona("persona.json")

# Sidebar: Show persona
with st.sidebar:
    st.header(" User Persona")
    if persona:
        if persona.get("habits"):
            st.subheader(" Habits")
            for h in persona["habits"]:
                st.write(f"• {h}")
        if persona.get("personality_traits"):
            st.subheader(" Personality")
            for p in persona["personality_traits"]:
                st.write(f"• {p}")
        if persona.get("communication_style"):
            st.subheader(" Communication Style")
            for c in persona["communication_style"]:
                st.write(f"• {c}")
        if persona.get("interests"):
            st.subheader(" Interests")
            for i in persona["interests"]:
                st.write(f"• {i}")
    else:
        st.warning("Persona not loaded. Run pipeline.py first!")

# Quick question buttons
st.subheader("Quick Questions:")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("What kind of person is this user?"):
        st.session_state.quick_query = "What kind of person is this user? Describe their personality."
with col2:
    if st.button("What are their habits?"):
        st.session_state.quick_query = "What are this user's habits and daily routines?"
with col3:
    if st.button("How do they talk?"):
        st.session_state.quick_query = "How does this user communicate? What is their communication style?"

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Handle quick question
if "quick_query" in st.session_state:
    query = st.session_state.quick_query
    del st.session_state.quick_query
    
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching conversations..."):
            answer = retrieve_and_answer(query, persona)
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Chat input
if query := st.chat_input("Ask anything about this user..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Searching conversations..."):
            answer = retrieve_and_answer(query, persona)
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})