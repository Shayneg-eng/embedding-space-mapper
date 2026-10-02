import os
from openai import OpenAI

# Initialize client — read the key from the environment, never hard-code it.
# Set it first:  export DEEPSEEK_API_KEY="sk-..."  (see .env.example)
api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise RuntimeError("DEEPSEEK_API_KEY is not set. See .env.example / the README.")
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

ANCHOR_FILE = 'anchor_words.txt'

def load_existing_words():
    """Load existing anchor words from file"""
    if os.path.exists(ANCHOR_FILE):
        with open(ANCHOR_FILE, 'r') as f:
            words = [line.strip() for line in f.readlines() if line.strip()]
        return set(words)
    return set()

def generate_anchor_words(existing_words, count=100):
    """Generate new anchor words using DeepSeek API"""
    
    # Create prompt with existing words for context
    existing_words_list = sorted(list(existing_words))
    existing_words_str = ", ".join(existing_words_list[-50:]) if existing_words_list else "None yet"
    
    prompt = f"""Generate exactly {count} unique, diverse English words that would be good for embedding space testing.

Requirements:
- Each word should be unique (no duplicates)
- Words should be diverse in meaning, length, and category
- Words should NOT be in this list (already used): {existing_words_str}
- One word per line
- No explanations or numbering, just the words

Generate {count} new words:"""
    
    print(f"Calling DeepSeek API to generate {count} words...")
    print(f"Existing words in database: {len(existing_words)}")
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates lists of English words. Return ONLY the words, one per line, with no numbering or explanations."},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    
    # Parse response
    response_text = response.choices[0].message.content
    words = [word.strip().lower() for word in response_text.split('\n') if word.strip()]
    
    return words

def remove_duplicates(new_words, existing_words):
    """Remove any words that already exist"""
    unique_new_words = []
    duplicates = []
    
    for word in new_words:
        if word in existing_words:
            duplicates.append(word)
        elif word not in unique_new_words:  # Also remove duplicates within the new batch
            unique_new_words.append(word)
    
    if duplicates:
        print(f"Found {len(duplicates)} duplicate words (already in database)")
    
    return unique_new_words

def save_words(words):
    """Append new words to anchor_words.txt"""
    if words:
        with open(ANCHOR_FILE, 'a') as f:
            for word in words:
                f.write(word + '\n')
        print(f"Saved {len(words)} new words to {ANCHOR_FILE}")
    else:
        print("No new unique words to save")

def main():
    print("="*60)
    print("ANCHOR WORDS GENERATOR (CONTINUOUS)")
    print("="*60)
    print("Press Ctrl+C to stop\n")
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration}")
            print(f"{'='*60}")
            
            # Load existing words
            existing_words = load_existing_words()
            print(f"Loaded {len(existing_words)} existing anchor words")
            
            # Generate new words
            new_words = generate_anchor_words(existing_words, count=100)
            print(f"Generated {len(new_words)} words from API")
            
            # Remove duplicates
            unique_words = remove_duplicates(new_words, existing_words)
            print(f"After deduplication: {len(unique_words)} unique new words")
            
            # Save to file
            save_words(unique_words)
            
            # Final stats
            final_words = load_existing_words()
            print(f"Final total words in {ANCHOR_FILE}: {len(final_words)}")
            
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("STOPPED BY USER")
        print("="*60)
        final_words = load_existing_words()
        print(f"Completed {iteration} iterations")
        print(f"Total words in {ANCHOR_FILE}: {len(final_words)}")

if __name__ == '__main__':
    main()
