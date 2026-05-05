"""
Run this file FIRST to process all conversations and build the system.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import json
import os
from src.preprocessor import load_conversations, get_message_batches
from src.topic_detector import detect_topic_changes
from src.checkpoint_builder import build_topic_checkpoints, build_message_checkpoints
from src.rag_engine import store_checkpoints
from src.persona_extractor import extract_persona, save_persona

def run_pipeline():
    print("=" * 60)
    print("STARTING RAG PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    messages = load_conversations("data/conversations.csv")
    
    # Step 2: Detect topics
    topic_segments = detect_topic_changes(messages, window_size=5, threshold=0.45)
    
    # Step 3: Build topic checkpoints (summaries)
    topic_checkpoints = build_topic_checkpoints(topic_segments)
    
    # Step 4: Build 100-message checkpoints
    message_batches = get_message_batches(messages, batch_size=100)
    message_checkpoints = build_message_checkpoints(message_batches)
    
    # Step 5: Store everything in vector database
    store_checkpoints(topic_checkpoints, message_checkpoints, messages)
    
    # Step 6: Extract persona
    persona = extract_persona(topic_checkpoints)
    save_persona(persona)
    
    # Step 7: Save checkpoints for reference
    with open("topic_checkpoints.json", "w") as f:
        for cp in topic_checkpoints:
            cp_save = {k: v for k, v in cp.items() if k != 'raw_text'}
            json.dump(cp_save, f)
            f.write("\n")
    
    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE!")
    print(f"   Topics found: {len(topic_checkpoints)}")
    print(f"   Message batches: {len(message_checkpoints)}")
    print(f"   Persona saved to: persona.json")
    print("=" * 60)
    print("\n  Now run: streamlit run app.py")

if __name__ == "__main__":
    run_pipeline()