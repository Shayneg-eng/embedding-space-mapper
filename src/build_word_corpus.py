import urllib.request
import json
import random

OUTPUT_FILE = 'data/word_corpus.txt'
NUM_WORDS = 25000

def download_word_list(n=25000):
    """Download n most common English words from multiple sources"""
    print(f"Downloading {n} most common English words...")
    
    words = []
    seen = set()
    
    # Start with manually curated high-frequency words (to ensure we get "I", "a", etc.)
    common = [
        'i', 'a', 'the', 'be', 'to', 'of', 'and', 'in', 'have', 'it',
        'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this',
        'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or',
        'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so',
        'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when',
        'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people',
        'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than',
        'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back',
        'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even',
        'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us', 'is',
        'was', 'are', 'been', 'being', 'did', 'does', 'doing', 'had', 'has', 'having',
        'am', 'might', 'must', 'should', 'shall', 'may', 'ought', 'house', 'world', 'place',
        'thing', 'life', 'hand', 'part', 'man', 'woman', 'child', 'water', 'food', 'love',
        'heart', 'mind', 'body', 'head', 'face', 'eye', 'ear', 'name', 'number', 'line',
        'right', 'high', 'different', 'small', 'large', 'next', 'good', 'little', 'own', 'other'
    ]
    for w in common:
        if w not in seen:
            words.append(w)
            seen.add(w)
    print(f"Added {len(words)} curated common words")
    
    try:
        # Try Peter Norvig's word frequency list (already in frequency order)
        print("Trying to download Peter Norvig word frequency list...")
        url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa-no-swears.txt"
        with urllib.request.urlopen(url, timeout=10) as response:
            downloaded = response.read().decode('utf-8').strip().split('\n')
        count = 0
        for w in downloaded:
            w_clean = w.lower().strip()
            if w_clean and w_clean not in seen:
                words.append(w_clean)
                seen.add(w_clean)
                count += 1
        print(f"Added {count} words from Peter Norvig list (frequency order preserved)")
    except Exception as e:
        print(f"Peter Norvig failed: {e}")
    
    try:
        # Try COCA corpus frequency list as backup
        print("Trying to download COCA corpus frequency list...")
        url = "https://www.english-corpora.org/coca/format/samples/5000_words.txt"
        with urllib.request.urlopen(url, timeout=10) as response:
            downloaded = response.read().decode('utf-8').strip().split('\n')
        # COCA format: word\tfrequency, so we need to extract just the word
        count = 0
        for line in downloaded:
            if '\t' in line:
                word = line.split('\t')[0].lower().strip()
            else:
                word = line.lower().strip()
            if word and len(word) >= 1 and word.isalpha() and word not in seen:
                words.append(word)
                seen.add(word)
                count += 1
        print(f"Added {count} words from COCA corpus")
    except Exception as e:
        print(f"COCA corpus failed: {e}")
    
    try:
        # Try Wikipedia frequency list
        print("Trying to download Wikipedia word frequency list...")
        url = "https://raw.githubusercontent.com/IlyaSemenov/wikipedia-word-frequency/master/results/en_50k.txt"
        with urllib.request.urlopen(url, timeout=10) as response:
            downloaded = response.read().decode('utf-8').strip().split('\n')
        count = 0
        for line in downloaded:
            if line.strip():
                word = line.split()[0].lower()
                if word.isalpha() and word not in seen:
                    words.append(word)
                    seen.add(word)
                    count += 1
        print(f"Added {count} words from Wikipedia frequency list")
    except Exception as e:
        print(f"Wikipedia frequency failed: {e}")
    
    # If still need more, get from dwyl's comprehensive list
    if len(words) < n:
        try:
            print(f"Need more words ({len(words)} < {n}), fetching from comprehensive list...")
            url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
            with urllib.request.urlopen(url, timeout=15) as response:
                downloaded = response.read().decode('utf-8').strip().split('\n')
            
            # Filter for words 6 characters or under
            additional = []
            for w in downloaded:
                w_clean = w.lower().strip()
                if w_clean.isalpha() and 1 <= len(w_clean) <= 6 and w_clean not in seen:
                    additional.append(w_clean)
            
            print(f"Found {len(additional)} words with 6 chars or under")
            
            # Shuffle to avoid alphabetical clustering
            random.shuffle(additional)
            
            # Add shuffled words
            for w in additional:
                if w not in seen:
                    words.append(w)
                    seen.add(w)
                    if len(words) >= n:
                        break
            
            print(f"Added words, total: {len(words)}")
        except Exception as e:
            print(f"Comprehensive list failed: {e}")
    
    # Final deduplication check
    deduplicated = []
    final_seen = set()
    for w in words:
        if w not in final_seen:
            deduplicated.append(w)
            final_seen.add(w)
    
    print(f"After deduplication: {len(deduplicated)} unique words")
    
    # Limit to n words
    return deduplicated[:n]

def save_corpus(words, filepath):
    """Save word list to file"""
    with open(filepath, 'w') as f:
        for word in words:
            f.write(word + '\n')
    print(f"Saved {len(words)} words to {filepath}")

def load_corpus(filepath):
    """Load word list from file"""
    with open(filepath, 'r') as f:
        words = [line.strip() for line in f if line.strip()]
    return words

def main():
    print("="*60)
    print("WORD CORPUS GENERATOR")
    print("="*60 + "\n")
    
    # Download words
    words = download_word_list(NUM_WORDS)
    
    if words is None:
        print("Failed to download word list")
        return
    
    # Save corpus
    save_corpus(words, OUTPUT_FILE)
    
    # Display sample
    print(f"\nFirst 50 words:")
    for i, word in enumerate(words[:50], 1):
        print(f"  {i:2d}. {word}")
    
    print(f"\nLast 50 words:")
    for i, word in enumerate(words[-50:], len(words)-49):
        print(f"  {i:5d}. {word}")
    
    print(f"\n{'='*60}")
    print(f"Total words: {len(words)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"File size: ~{len(words) * 10 / 1024:.1f} KB")

if __name__ == '__main__':
    main()
