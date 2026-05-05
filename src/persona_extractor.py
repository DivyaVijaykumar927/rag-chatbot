import json
import os
import re
import anthropic
from dotenv import load_dotenv

load_dotenv()


def extract_persona_simple(topic_checkpoints):
    """
    Extract persona using simple keyword analysis - completely FREE.
    No API needed.
    """
    print("\nExtracting user persona (free mode)...")
    
    # Combine all raw text
    all_text = " ".join([
        cp.get('raw_text', '') for cp in topic_checkpoints
    ]).lower()
    
    # --- Habits Detection ---
    habits = []
    if any(w in all_text for w in ['late night', 'midnight', 'cant sleep', "can't sleep", '2am', '3am', '1am']):
        habits.append("Late sleeper / stays up late")
    if any(w in all_text for w in ['skip breakfast', 'no breakfast', 'not eating', 'forgot to eat']):
        habits.append("Skips meals sometimes")
    if any(w in all_text for w in ['coffee', 'tea', 'chai']):
        habits.append("Drinks coffee/tea regularly")
    if any(w in all_text for w in ['gym', 'workout', 'exercise', 'running', 'walk']):
        habits.append("Has exercise habits")
    if any(w in all_text for w in ['netflix', 'series', 'binge', 'episode', 'movie']):
        habits.append("Watches shows/movies regularly")
    if any(w in all_text for w in ['phone', 'scroll', 'instagram', 'social media']):
        habits.append("Uses social media frequently")
    
    # --- Personal Facts Detection ---
    personal_facts = []
    if any(w in all_text for w in ['my mom', 'my mother', 'my dad', 'my father', 'my parents']):
        personal_facts.append("Has parents mentioned in conversations")
    if any(w in all_text for w in ['my sister', 'my brother', 'my sibling']):
        personal_facts.append("Has siblings")
    if any(w in all_text for w in ['my boyfriend', 'my girlfriend', 'my partner', 'my husband', 'my wife']):
        personal_facts.append("In a romantic relationship")
    if any(w in all_text for w in ['college', 'university', 'exam', 'study', 'semester']):
        personal_facts.append("Currently studying or recently studied")
    if any(w in all_text for w in ['job', 'work', 'office', 'boss', 'salary', 'internship']):
        personal_facts.append("Working or looking for work")
    if any(w in all_text for w in ['my dog', 'my cat', 'my pet']):
        personal_facts.append("Has a pet")
    
    # --- Personality Traits Detection ---
    personality = []
    if any(w in all_text for w in ['lol', 'haha', 'hehe', 'funny', 'joke', 'laugh']):
        personality.append("Has a good sense of humor")
    if any(w in all_text for w in ['anxious', 'anxiety', 'stress', 'worried', 'overthink']):
        personality.append("Tends to be anxious or overthink")
    if any(w in all_text for w in ['miss', 'feel bad', 'sad', 'upset', 'cry', 'emotional']):
        personality.append("Emotionally expressive")
    if any(w in all_text for w in ['love', 'care', 'support', 'help', 'kind']):
        personality.append("Caring and supportive")
    if any(w in all_text for w in ['honest', 'truth', 'real', 'genuine']):
        personality.append("Values honesty")
    if any(w in all_text for w in ['idk', 'not sure', 'confused', 'lost']):
        personality.append("Sometimes indecisive or uncertain")
    
    # --- Communication Style Detection ---
    comm_style = []
    
    # Check average message length from raw texts
    all_messages_text = [cp.get('raw_text', '') for cp in topic_checkpoints]
    lines = []
    for t in all_messages_text:
        lines.extend(t.split('\n'))
    
    if lines:
        avg_len = sum(len(l) for l in lines) / len(lines)
        if avg_len < 30:
            comm_style.append("Sends short, brief messages")
        elif avg_len > 80:
            comm_style.append("Writes long, detailed messages")
        else:
            comm_style.append("Moderate message length")
    
    emoji_count = sum(1 for ch in all_text if ord(ch) > 127000)
    if emoji_count > 20:
        comm_style.append("Uses emojis frequently")
    
    if any(w in all_text for w in ['haha', 'lmao', 'lol', 'xD', 'XD']):
        comm_style.append("Uses casual/informal language")
    
    if '?' in all_text and all_text.count('?') > 50:
        comm_style.append("Asks many questions, curious nature")
    
    # --- Interests Detection ---
    interests = []
    if any(w in all_text for w in ['music', 'song', 'playlist', 'singer', 'album']):
        interests.append("Music")
    if any(w in all_text for w in ['movie', 'film', 'cinema', 'netflix', 'series']):
        interests.append("Movies and TV shows")
    if any(w in all_text for w in ['food', 'eat', 'cook', 'recipe', 'restaurant']):
        interests.append("Food and cooking")
    if any(w in all_text for w in ['travel', 'trip', 'visit', 'place', 'tour']):
        interests.append("Travel")
    if any(w in all_text for w in ['cricket', 'football', 'sport', 'match', 'team']):
        interests.append("Sports")
    if any(w in all_text for w in ['book', 'read', 'novel', 'author']):
        interests.append("Reading")
    if any(w in all_text for w in ['game', 'gaming', 'play', 'pubg', 'fifa']):
        interests.append("Gaming")
    
    persona = {
        "habits": habits if habits else ["Not enough data to determine habits"],
        "personal_facts": personal_facts if personal_facts else ["Not enough data"],
        "personality_traits": personality if personality else ["Not enough data"],
        "communication_style": comm_style if comm_style else ["Not enough data"],
        "interests": interests if interests else ["Not enough data"],
        "emotional_patterns": ["Expressive in conversations"] if any(
            w in all_text for w in ['feel', 'emotion', 'heart', 'mood']
        ) else ["Neutral emotional expression"]
    }
    
    print("Persona extracted successfully!")
    for key, val in persona.items():
        print(f"   {key}: {len(val)} items found")
    
    return persona


def extract_persona(topic_checkpoints):
    """Main function - tries API first, falls back to simple."""
    key = os.getenv("ANTHROPIC_API_KEY")
    
    if key:
        try:
            client = anthropic.Anthropic(api_key=key)
            
            # Use only first 20 topic summaries to save credits
            sample_summaries = "\n\n".join([
                f"Topic {cp['topic_number']}: {cp['summary']}"
                for cp in topic_checkpoints[:20]
            ])
            
            prompt = f"""Analyze these conversation summaries and extract user persona.
Return ONLY a JSON object, no other text:

{{
  "habits": ["list habits"],
  "personal_facts": ["list facts"],
  "personality_traits": ["list traits"],
  "communication_style": ["list styles"],
  "interests": ["list interests"],
  "emotional_patterns": ["list patterns"]
}}

Summaries:
{sample_summaries}"""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.content[0].text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            return json.loads(text)
        
        except Exception as e:
            print(f"API persona extraction failed: {e}")
            print("Falling back to simple extraction...")
            return extract_persona_simple(topic_checkpoints)
    else:
        return extract_persona_simple(topic_checkpoints)


def save_persona(persona, path="persona.json"):
    with open(path, 'w') as f:
        json.dump(persona, f, indent=2)
    print(f"Persona saved to {path}")


def load_persona(path="persona.json"):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None