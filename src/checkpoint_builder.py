import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

# We'll try Claude API, but fall back to simple summarization if no credits
def get_client():
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        return anthropic.Anthropic(api_key=key)
    return None

def simple_summary(messages, max_sentences=5):
    """
    Free summarization - no API needed.
    Takes the first + last few messages and key middle ones.
    """
    if not messages:
        return "Empty segment."
    
    texts = [f"{m['sender']}: {m['text']}" for m in messages]
    total = len(texts)
    
    # Pick representative messages
    selected = []
    
    # First 2 messages
    selected.extend(texts[:2])
    
    # Middle message
    if total > 4:
        selected.append(texts[total // 2])
    
    # Last 2 messages
    if total > 2:
        selected.extend(texts[-2:])
    
    # Join them as summary
    summary = " | ".join(selected)
    
    # Truncate if too long
    if len(summary) > 500:
        summary = summary[:500] + "..."
    
    return summary


def generate_summary(text, summary_type="topic"):
    """
    Try Claude API first. If no credits, use simple summary.
    """
    client = get_client()
    
    if client:
        try:
            if summary_type == "topic":
                prompt = f"""Summarize this conversation segment in 3-4 sentences.
Focus on: main subject, key points, emotions shown.

Conversation:
{text[:2000]}

Summary:"""
            else:
                prompt = f"""Summarize these messages in 4-5 sentences.
Include: main topics, tone, important events.

Conversation:
{text[:2000]}

Summary:"""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        
        except Exception as e:
            print(f"     API failed ({str(e)[:50]}), using simple summary...")
            return text[:400] + "..." if len(text) > 400 else text
    else:
        return text[:400] + "..." if len(text) > 400 else text


def messages_to_text(messages):
    """Convert list of message dicts to readable text."""
    lines = []
    for msg in messages:
        lines.append(f"{msg['sender']}: {msg['text']}")
    return "\n".join(lines)


def build_topic_checkpoints(topic_segments):
    """
    Build checkpoints for each topic segment.
    Uses simple summarization - no API needed!
    """
    print(f"\nBuilding topic checkpoints for {len(topic_segments)} topics...")
    checkpoints = []
    
    for segment in topic_segments:
        # Use simple summary (free, no API)
        summary = simple_summary(segment['messages'])
        
        checkpoint = {
            'type': 'topic',
            'topic_number': segment['topic_number'],
            'start_index': segment['start_index'],
            'end_index': segment['end_index'],
            'message_count': len(segment['messages']),
            'summary': summary,
            'raw_text': messages_to_text(segment['messages'])[:1000]
        }
        checkpoints.append(checkpoint)
        
        # Print progress every 50 topics
        if segment['topic_number'] % 50 == 0:
            print(f"   Done {segment['topic_number']}/{len(topic_segments)} topics...")
    
    print(f"Topic checkpoints built: {len(checkpoints)}")
    return checkpoints


def build_message_checkpoints(message_batches):
    """
    Build checkpoints for every 100 messages.
    Uses simple summarization - no API needed!
    """
    print(f"\nBuilding 100-message checkpoints...")
    checkpoints = []
    
    for batch in message_batches:
        summary = simple_summary(batch['messages'])
        
        checkpoint = {
            'type': '100-message',
            'batch_number': batch['batch_number'],
            'start_index': batch['start_index'],
            'end_index': batch['end_index'],
            'message_count': len(batch['messages']),
            'summary': summary,
            'raw_text': messages_to_text(batch['messages'])[:1000]
        }
        checkpoints.append(checkpoint)
    
    print(f"Message checkpoints built: {len(checkpoints)}")
    return checkpoints