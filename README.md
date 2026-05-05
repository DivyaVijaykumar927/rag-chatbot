# RAG Conversation Analysis Chatbot

A system that reads through chat conversations, understands what topics were discussed, builds a profile of the user, and lets you ask questions about them through a chatbot interface.

---

## Setup and Running

Make sure Python 3.10 is installed. Then:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the root folder and add:
```
ANTHROPIC_API_KEY=your_key_here
```

Put your CSV file inside the `data/` folder and name it `conversations.csv`

Then run the pipeline first (this processes everything):
```bash
python pipeline.py
```

This will take a few minutes depending on how large your CSV is. Once it finishes, start the chatbot:
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## How Topic Detection Works

The core idea is simple — if two messages are talking about completely different things, their meaning vectors will point in different directions.

Every message gets converted into a 384-dimensional vector using the `all-MiniLM-L6-v2` model from sentence-transformers. This runs locally, no API needed.

For each new message, I calculate the cosine similarity between it and the 5 messages before it. If that similarity score drops below 0.45, it means the conversation has shifted to a new topic — so a new topic checkpoint is created.

This gave us 511 topic segments across 10,000+ messages, which felt realistic for a long conversation spanning many days.

---

## How the RAG Retrieval Works

All topic summaries, 100-message checkpoints, and raw message chunks are stored in ChromaDB (a local vector database).

When you type a question:
1. The question gets converted to an embedding
2. ChromaDB does a similarity search to find the 5 most relevant topic summaries
3. It also finds the 5 most relevant raw message chunks
4. Both get combined into a context block
5. That context is sent to Claude API which generates the final answer

So the system never answers from memory — it always retrieves actual conversation data first.

---

## How the Persona Gets Built

Instead of asking an LLM to guess, the persona extraction scans the actual conversation text for real signals:

- **Habits** — looks for patterns like "can't sleep", "2am", "skipped lunch", "went to gym"
- **Personal facts** — picks up mentions of "my sister", "my boyfriend", "my job", "college exam"
- **Personality** — detects tone from words like "lol", "honestly", "I always overthink", "I feel bad"
- **Communication style** — measures average message length, emoji frequency, use of informal words

Everything gets saved to `persona.json` in a clean structured format.

---

## Live Demo

Hosted on Streamlit Cloud:
`https://rag-chatbot-divya.streamlit.app/`

---

## Tech Used

- sentence-transformers — for converting messages to vectors locally
- ChromaDB — stores and searches vectors
- Streamlit — the chatbot UI





## VIDEO 
https://github.com/user-attachments/assets/022870b3-04a6-49d4-a421-d9dc8602cb85


## IMAGES
<img width="1913" height="684" alt="IMAGEE_3" src="https://github.com/user-attachments/assets/7d985f19-18fe-4c5c-a2b1-87568dd472c1" />
<img width="1805" height="811" alt="IMAGEE_2" src="https://github.com/user-attachments/assets/33021490-afc6-44f3-a566-25adbf063596" />
<img width="1906" height="843" alt="IMAGEE_1" src="https://github.com/user-attachments/assets/e4f23719-c9eb-400f-97ed-93bfa0daff52" />



- Anthropic Claude API — generates final answers from retrieved context
- Python 3.10
