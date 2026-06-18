# RAG Conversation Analysis Chatbot

A system that reads through chat conversations, understands what topics were discussed, builds a profile of the user, and lets you ask questions about them through a chatbot interface.

---

## Setup and Running

Make sure Python 3.10 is installed. Then:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the root folder and add:

```env
ANTHROPIC_API_KEY=your_key_here
```

Put your CSV file inside the `data/` folder and name it `conversations.csv`

Then run the pipeline first:

```bash
python pipeline.py
```

Once processing is complete, start the chatbot:

```bash
streamlit run app.py
```

Open your browser at:

```text
http://localhost:8501
```

---

## How Topic Detection Works

Every message is converted into a 384-dimensional embedding using the `all-MiniLM-L6-v2` model from Sentence-Transformers.

For each incoming message, cosine similarity is calculated against the previous 5 messages. When similarity drops below 0.45, a topic boundary is detected and a new topic segment is created.

This generated 511+ topic segments across 10,000+ conversation messages.

---

## How the RAG Retrieval Works

All topic summaries, checkpoints, and conversation chunks are stored in ChromaDB.

When a user asks a question:

1. The query is converted into an embedding.
2. ChromaDB retrieves the top-5 most relevant topic summaries.
3. ChromaDB retrieves the top-5 most relevant conversation chunks.
4. Retrieved context is combined.
5. Claude API generates a grounded response using retrieved information.

This ensures responses are generated from retrieved conversation data rather than relying on model memory.

---

## How Persona Extraction Works

The persona extraction pipeline analyzes actual conversation content to identify:

* Habits and routines
* Personal facts and relationships
* Personality traits
* Communication style
* Message behavior patterns

Extracted information is stored in a structured `persona.json` file.

---

## Cloud Deployment

The application was deployed on Microsoft Azure App Service using GitHub Actions CI/CD.

Deployment workflow:

1. Source code hosted on GitHub.
2. Azure Deployment Center connected to the GitHub repository.
3. GitHub Actions automatically builds and deploys the application.
4. Azure App Service hosts the chatbot and manages runtime infrastructure.
5. Environment variables such as `ANTHROPIC_API_KEY` are securely configured through Azure App Service settings.

This enables cloud-based hosting, automated deployment, and scalable access to the RAG application.

---

## Live Demo

Streamlit Cloud:

https://rag-chatbot-divya.streamlit.app/

Azure App Service:

Deployed using Microsoft Azure App Service with GitHub Actions CI/CD.

---

## Tech Stack

* Python 3.10
* Streamlit
* Sentence-Transformers
* ChromaDB
* Scikit-Learn
* NumPy
* Pandas
* Anthropic Claude API
* Microsoft Azure App Service
* GitHub Actions CI/CD

---

## Features

* Semantic Topic Segmentation
* Retrieval-Augmented Generation (RAG)
* Persona Extraction
* Vector Search using ChromaDB
* Context-Aware Question Answering
* Azure Cloud Deployment
* Automated CI/CD Pipeline
