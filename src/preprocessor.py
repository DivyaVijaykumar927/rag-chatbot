import pandas as pd
import os

def load_conversations(csv_path):
    """
    Load the CSV file and prepare messages in chronological order.
    Returns a list of message dictionaries.
    """
    print(f"📂 Loading CSV from: {csv_path}")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Print column names so you can see what's in your CSV
    print(f"📋 Columns found: {list(df.columns)}")
    print(f"📊 Total rows: {len(df)}")
    print(f"🔍 First few rows:\n{df.head(3)}")
    
    # IMPORTANT: Adjust these column names to match YOUR CSV
    # Common column names to look for:
    possible_message_cols = ['message', 'text', 'content', 'Message', 'Text']
    possible_sender_cols = ['sender', 'user', 'from', 'Sender', 'User']
    possible_time_cols = ['timestamp', 'time', 'date', 'Timestamp', 'Date']
    
    # Find which columns exist in your CSV
    message_col = next((c for c in possible_message_cols if c in df.columns), df.columns[0])
    sender_col = next((c for c in possible_sender_cols if c in df.columns), None)
    time_col = next((c for c in possible_time_cols if c in df.columns), None)
    
    print(f" Using message column: '{message_col}'")
    print(f" Using sender column: '{sender_col}'")
    print(f" Using time column: '{time_col}'")
    
    # Sort by time if time column exists (chronological order)
    if time_col:
        df = df.sort_values(by=time_col).reset_index(drop=True)
        print("Sorted chronologically")
    
    # Build list of messages
    messages = []
    for idx, row in df.iterrows():
        msg = {
            'index': idx,
            'text': str(row[message_col]).strip(),
            'sender': str(row[sender_col]) if sender_col else 'unknown',
            'timestamp': str(row[time_col]) if time_col else str(idx),
        }
        # Skip empty messages
        if msg['text'] and msg['text'].lower() != 'nan':
            messages.append(msg)
    
    print(f" Loaded {len(messages)} valid messages")
    return messages


def get_message_batches(messages, batch_size=100):
    """
    Split messages into batches of 100 for checkpoint creation.
    """
    batches = []
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        batches.append({
            'batch_number': (i // batch_size) + 1,
            'start_index': i,
            'end_index': min(i + batch_size - 1, len(messages) - 1),
            'messages': batch
        })
    return batches


# Test this file independently
if __name__ == "__main__":
    messages = load_conversations("data/conversations.csv")
    batches = get_message_batches(messages)
    print(f"\n Created {len(batches)} batches of 100 messages")