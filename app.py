import streamlit as st
import json
import os
from src.persona_extractor import load_persona

# Page setup
st.set_page_config(page_title="Conversation RAG Chatbot", page_icon="robot", layout="wide")
st.title("Conversation Analysis Chatbot")
st.markdown("Ask me anything about the user based on their conversation history!")

# Load persona
persona = load_persona("persona.json")

# Check if ChromaDB exists on this machine
chroma_exists = os.path.exists("./chroma_db")

# Sidebar: Show persona
with st.sidebar:
    st.header("User Persona")
    if persona:
        if persona.get("habits"):
            st.subheader("Habits")
            for h in persona["habits"]:
                st.write(f"- {h}")
        if persona.get("personality_traits"):
            st.subheader("Personality")
            for p in persona["personality_traits"]:
                st.write(f"- {p}")
        if persona.get("communication_style"):
            st.subheader("Communication Style")
            for c in persona["communication_style"]:
                st.write(f"- {c}")
        if persona.get("interests"):
            st.subheader("Interests")
            for i in persona["interests"]:
                st.write(f"- {i}")
        if persona.get("personal_facts"):
            st.subheader("Personal Facts")
            for f in persona["personal_facts"]:
                st.write(f"- {f}")
    else:
        st.warning("Persona not loaded. Run pipeline.py first!")


def answer_from_persona_only(query, persona_data):
    """
    Answer questions using ONLY persona data.
    No ChromaDB needed - works on cloud deployment!
    """
    if not persona_data:
        return "No persona data available."

    query_lower = query.lower()

    # Personality questions
    if any(w in query_lower for w in ['person', 'personality', 'kind of', 'who', 'describe', 'character']):
        traits = persona_data.get('personality_traits', [])
        emotional = persona_data.get('emotional_patterns', [])
        interests = persona_data.get('interests', [])
        style = persona_data.get('communication_style', [])

        answer = "Based on conversation analysis, here is this user's profile:\n\n"
        if traits:
            answer += "**Personality Traits:**\n"
            for t in traits:
                answer += f"  - {t}\n"
        if emotional:
            answer += "\n**Emotional Patterns:**\n"
            for e in emotional:
                answer += f"  - {e}\n"
        if interests:
            answer += "\n**Interests:**\n"
            for i in interests:
                answer += f"  - {i}\n"
        if style:
            answer += "\n**Communication Style:**\n"
            for s in style:
                answer += f"  - {s}\n"
        return answer

    # Habits questions
    if any(w in query_lower for w in ['habit', 'routine', 'daily', 'sleep', 'eat', 'lifestyle', 'morning', 'night']):
        habits = persona_data.get('habits', [])
        facts = persona_data.get('personal_facts', [])

        answer = "Based on conversation analysis, here are this user's habits:\n\n"
        if habits:
            answer += "**Daily Habits:**\n"
            for h in habits:
                answer += f"  - {h}\n"
        if facts:
            answer += "\n**Personal Facts:**\n"
            for f in facts:
                answer += f"  - {f}\n"
        return answer

    # Communication style questions
    if any(w in query_lower for w in ['talk', 'speak', 'communicate', 'message', 'style', 'tone', 'text', 'write']):
        style = persona_data.get('communication_style', [])
        traits = persona_data.get('personality_traits', [])

        answer = "Based on conversation analysis, here is how this user communicates:\n\n"
        if style:
            answer += "**Communication Style:**\n"
            for s in style:
                answer += f"  - {s}\n"
        if traits:
            answer += "\n**Related Personality Traits:**\n"
            for t in traits[:3]:
                answer += f"  - {t}\n"
        return answer

    # Interests questions
    if any(w in query_lower for w in ['interest', 'hobby', 'enjoy', 'passion', 'like', 'love']):
        interests = persona_data.get('interests', [])
        answer = "Based on conversation analysis, here are this user's interests:\n\n"
        if interests:
            for i in interests:
                answer += f"  - {i}\n"
        return answer

    # General fallback - show everything
    answer = f"Here is what I know about this user related to '{query}':\n\n"
    for key, values in persona_data.items():
        if values and isinstance(values, list):
            label = key.replace('_', ' ').title()
            answer += f"**{label}:**\n"
            for v in values:
                answer += f"  - {v}\n"
            answer += "\n"
    return answer


def get_answer(query, persona_data):
    """
    Try RAG first if ChromaDB exists locally.
    Fall back to persona-only mode if on cloud or DB missing.
    """
    if chroma_exists:
        try:
            from src.rag_engine import retrieve_and_answer
            return retrieve_and_answer(query, persona_data)
        except Exception as e:
            st.warning("RAG database unavailable, using persona data only.")
            return answer_from_persona_only(query, persona_data)
    else:
        return answer_from_persona_only(query, persona_data)


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

# Handle quick question button clicks
if "quick_query" in st.session_state:
    query = st.session_state.quick_query
    del st.session_state.quick_query

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching conversations..."):
            answer = get_answer(query, persona)
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Chat input box at bottom
if query := st.chat_input("Ask anything about this user..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching conversations..."):
            answer = get_answer(query, persona)
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})