import chromadb
import json
import os
from sentence_transformers import SentenceTransformer
import anthropic
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')

# Only create client if API key exists
api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=api_key) if api_key else None

# Create ChromaDB (a local database that stores vectors)
chroma_client = chromadb.PersistentClient(path="./chroma_db")


def setup_collections():
    """Create or get database collections."""
    try:
        chroma_client.delete_collection("topic_checkpoints")
        chroma_client.delete_collection("message_checkpoints")
        chroma_client.delete_collection("raw_messages")
    except:
        pass

    topic_col = chroma_client.create_collection("topic_checkpoints")
    msg_col = chroma_client.create_collection("message_checkpoints")
    raw_col = chroma_client.create_collection("raw_messages")

    return topic_col, msg_col, raw_col


def store_checkpoints(topic_checkpoints, message_checkpoints, messages):
    """Store all data in ChromaDB for later retrieval."""
    print("\nStoring data in vector database...")

    topic_col, msg_col, raw_col = setup_collections()

    # Store topic checkpoints
    print(f"   Storing {len(topic_checkpoints)} topic checkpoints...")
    for cp in topic_checkpoints:
        embedding = model.encode([cp['summary']])[0].tolist()
        topic_col.add(
            ids=[f"topic_{cp['topic_number']}"],
            embeddings=[embedding],
            documents=[cp['summary']],
            metadatas=[{
                'topic_number': cp['topic_number'],
                'start_index': cp['start_index'],
                'end_index': cp['end_index'],
                'raw_text': cp['raw_text'][:500]
            }]
        )

    # Store 100-message checkpoints
    print(f"   Storing {len(message_checkpoints)} message checkpoints...")
    for cp in message_checkpoints:
        embedding = model.encode([cp['summary']])[0].tolist()
        msg_col.add(
            ids=[f"batch_{cp['batch_number']}"],
            embeddings=[embedding],
            documents=[cp['summary']],
            metadatas=[{
                'batch_number': cp['batch_number'],
                'start_index': cp['start_index'],
                'end_index': cp['end_index']
            }]
        )

    # Store raw messages in chunks of 10
    print(f"   Storing raw messages in chunks...")
    chunk_size = 10
    for i in range(0, len(messages), chunk_size):
        chunk = messages[i:i + chunk_size]
        chunk_text = "\n".join([f"{m['sender']}: {m['text']}" for m in chunk])
        embedding = model.encode([chunk_text])[0].tolist()
        raw_col.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk_text],
            metadatas=[{
                'start_index': i,
                'end_index': min(i + chunk_size - 1, len(messages) - 1)
            }]
        )

    print("All data stored in ChromaDB!")
    return topic_col, msg_col, raw_col


def build_local_answer(query, topic_docs, raw_docs, persona_data):
    """
    Build a full answer using only local retrieved data.
    No API credits needed at all.
    """
    query_lower = query.lower()

    # --- Personality / Person questions ---
    if any(w in query_lower for w in ['person', 'personality', 'kind of', 'who is', 'describe', 'like']):
        if persona_data:
            traits = persona_data.get('personality_traits', [])
            style = persona_data.get('communication_style', [])
            interests = persona_data.get('interests', [])
            emotional = persona_data.get('emotional_patterns', [])

            answer = "Based on the conversation analysis, here is a description of this user:\n\n"

            if traits:
                answer += "**Personality Traits:**\n"
                for t in traits:
                    answer += f"  - {t}\n"
                answer += "\n"

            if emotional:
                answer += "**Emotional Patterns:**\n"
                for e in emotional:
                    answer += f"  - {e}\n"
                answer += "\n"

            if interests:
                answer += "**Interests:**\n"
                for i in interests:
                    answer += f"  - {i}\n"
                answer += "\n"

            if style:
                answer += "**How they communicate:**\n"
                for s in style:
                    answer += f"  - {s}\n"
                answer += "\n"

            answer += "**Evidence from conversations:**\n"
            for doc in topic_docs[:3]:
                answer += f"  - {doc[:200]}\n"

            return answer

    # --- Habits questions ---
    if any(w in query_lower for w in ['habit', 'routine', 'daily', 'sleep', 'eat', 'morning', 'night', 'lifestyle']):
        if persona_data:
            habits = persona_data.get('habits', [])
            facts = persona_data.get('personal_facts', [])

            answer = "Based on the conversation analysis, here are this user's habits:\n\n"

            if habits:
                answer += "**Daily Habits:**\n"
                for h in habits:
                    answer += f"  - {h}\n"
                answer += "\n"

            if facts:
                answer += "**Personal Facts:**\n"
                for f in facts:
                    answer += f"  - {f}\n"
                answer += "\n"

            answer += "**Relevant conversation excerpts:**\n"
            for doc in raw_docs[:3]:
                answer += f"  - {doc[:200]}\n"

            return answer

    # --- Communication style questions ---
    if any(w in query_lower for w in ['talk', 'speak', 'communicate', 'message', 'style', 'tone', 'write', 'text']):
        if persona_data:
            style = persona_data.get('communication_style', [])
            traits = persona_data.get('personality_traits', [])

            answer = "Based on the conversation analysis, here is how this user communicates:\n\n"

            if style:
                answer += "**Communication Style:**\n"
                for s in style:
                    answer += f"  - {s}\n"
                answer += "\n"

            if traits:
                answer += "**Related Personality Traits:**\n"
                for t in traits:
                    answer += f"  - {t}\n"
                answer += "\n"

            answer += "**Sample conversation excerpts:**\n"
            for doc in raw_docs[:4]:
                answer += f"\n---\n{doc[:250]}\n"

            return answer

    # --- Interests questions ---
    if any(w in query_lower for w in ['interest', 'like', 'love', 'hobby', 'enjoy', 'passion']):
        if persona_data:
            interests = persona_data.get('interests', [])

            answer = "Based on the conversation analysis, here are this user's interests:\n\n"

            if interests:
                answer += "**Interests & Hobbies:**\n"
                for i in interests:
                    answer += f"  - {i}\n"
                answer += "\n"

            answer += "**Relevant conversation excerpts:**\n"
            for doc in raw_docs[:3]:
                answer += f"  - {doc[:200]}\n"

            return answer

    # --- General / Fallback answer ---
    answer = f"Here is what I found related to your question: '{query}'\n\n"

    answer += "**Relevant Topics Discussed:**\n"
    for doc in topic_docs[:4]:
        answer += f"  - {doc[:200]}\n"

    answer += "\n**Relevant Conversation Excerpts:**\n"
    for doc in raw_docs[:4]:
        answer += f"\n---\n{doc[:250]}\n"

    if persona_data:
        answer += "\n**User Profile Summary:**\n"
        traits = persona_data.get('personality_traits', [])
        habits = persona_data.get('habits', [])
        interests = persona_data.get('interests', [])
        if traits:
            answer += f"  Personality: {', '.join(traits[:3])}\n"
        if habits:
            answer += f"  Habits: {', '.join(habits[:3])}\n"
        if interests:
            answer += f"  Interests: {', '.join(interests[:3])}\n"

    return answer


def retrieve_and_answer(query, persona_data=None):
    """
    Main RAG function: Given a question, retrieve relevant info and answer.
    Works with OR without Anthropic API credits.
    """
    print(f"\nProcessing query: {query}")

    # Load collections
    topic_col = chroma_client.get_collection("topic_checkpoints")
    raw_col = chroma_client.get_collection("raw_messages")

    # Get query embedding
    query_embedding = model.encode([query])[0].tolist()

    # Retrieve top 5 relevant topic summaries
    topic_results = topic_col.query(
        query_embeddings=[query_embedding], n_results=5)

    # Retrieve top 5 relevant message chunks
    raw_results = raw_col.query(
        query_embeddings=[query_embedding], n_results=5)

    topic_docs = topic_results['documents'][0]
    raw_docs = raw_results['documents'][0]

    # Try Claude API first (if credits available)
    if client:
        try:
            context = "=== RELEVANT TOPIC SUMMARIES ===\n"
            context += "\n".join([f"- {doc}" for doc in topic_docs])
            context += "\n\n=== RELEVANT CONVERSATION CHUNKS ===\n"
            context += "\n---\n".join(raw_docs)

            if persona_data:
                context += f"\n\n=== USER PERSONA ===\n{json.dumps(persona_data, indent=2)}"

            prompt = f"""You are an assistant analyzing a user's conversation history.
Use ONLY the information below to answer. Be specific and helpful.

{context}

Question: {query}

Answer:"""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()

        except Exception as e:
            print(f"API failed, switching to local answer: {e}")

    # Fall back to local answer (completely free, no API needed)
    return build_local_answer(query, topic_docs, raw_docs, persona_data)