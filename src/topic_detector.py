from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the embedding model once (this runs when file is imported)
print(" Loading sentence embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print(" Embedding model loaded!")


def get_embedding(text):
    """Convert text to a vector (list of numbers representing meaning)."""
    return model.encode([text])[0]


def detect_topic_changes(messages, window_size=5, threshold=0.45):
    """
    Detect where topics change in the conversation.
    
    How it works:
    - We look at every message's meaning (as a vector/embedding)
    - We compare it to the previous few messages
    - If the similarity drops below threshold → topic changed!
    
    window_size: How many previous messages to compare against
    threshold: If similarity < this number, it's a new topic (0 to 1)
    """
    print(f"\n Detecting topic changes in {len(messages)} messages...")
    print(f"   Window size: {window_size}, Threshold: {threshold}")
    
    if len(messages) < 2:
        return [{'topic_number': 1, 'start_index': 0, 
                 'end_index': len(messages)-1, 'messages': messages}]
    
    # Step 1: Get embeddings for all messages
    print("   Computing embeddings for all messages (may take a moment)...")
    texts = [msg['text'] for msg in messages]
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"   Got {len(embeddings)} embeddings")
    
    # Step 2: Find change points
    change_points = [0]  # First message always starts a topic
    
    for i in range(window_size, len(embeddings)):
        # Get current message embedding
        current = embeddings[i].reshape(1, -1)
        
        # Get window of previous messages
        window_start = max(0, i - window_size)
        previous_window = embeddings[window_start:i]
        
        # Compute similarity between current and each in window
        similarities = cosine_similarity(current, previous_window)[0]
        avg_similarity = np.mean(similarities)
        
        # If similarity is low → topic changed!
        if avg_similarity < threshold:
            # Make sure this isn't too close to the last change point
            if i - change_points[-1] >= 10:  # At least 10 messages per topic
                change_points.append(i)
                print(f"    Topic change detected at message {i} "
                      f"(similarity: {avg_similarity:.3f})")
    
    # Step 3: Build topic segments
    topic_segments = []
    for i, start in enumerate(change_points):
        end = change_points[i + 1] - 1 if i + 1 < len(change_points) else len(messages) - 1
        
        segment = {
            'topic_number': i + 1,
            'start_index': start,
            'end_index': end,
            'messages': messages[start:end + 1]
        }
        topic_segments.append(segment)
        print(f"   Topic {i+1}: messages {start}–{end} "
              f"({end - start + 1} messages)")
    
    print(f"\n Found {len(topic_segments)} topics total")
    return topic_segments


# Test this file independently
if __name__ == "__main__":
    # Test with fake messages
    test_messages = [
        {'index': i, 'text': text, 'sender': 'user', 'timestamp': str(i)}
        for i, text in enumerate([
            "I love eating pizza", "Pizza is my favorite food",
            "Had pasta for dinner", "Food is amazing",
            "Let's talk about movies", "I watched Inception yesterday",
            "Christopher Nolan is brilliant", "Have you seen Interstellar?",
            "My dog is sick today", "Took him to the vet",
            "The vet said he needs rest", "Poor doggo"
        ])
    ]
    segments = detect_topic_changes(test_messages, window_size=2, threshold=0.4)
    print(f"Detected {len(segments)} topics")