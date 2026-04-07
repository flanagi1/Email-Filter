import os
import re
from collections import Counter

def remove_stopword(fp):
    junk = set()
    if not os.path.exists(fp):
        return junk
    with open(fp, 'r') as fl:
        for line in fl:
            word = line.strip().lower()
            if word:
                junk.add(word)
    return junk

def count_word_frequencies(folder_path, junk):
    """
    Counts frequencies for a single specific folder.
    Now accepts the 'junk' set directly to avoid reloading it every time.
    """
    cnt = Counter()
    
    # List files in the specific sub-category folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    words = re.split(r'[^a-zA-Z0-9]+', line)
                    filt = [word.lower() for word in words if word and word.lower() not in junk]
                    cnt.update(filt)
    return cnt

def process_all_categories(root_folder, stop_file):
    # 1. Load stopwords once
    junk = remove_stopword(stop_file)
    
    # 2. This dictionary will hold: {'school': Counter(), 'spam': Counter(), ...}
    category_data = {}

    # 3. Iterate through items in the 'emails' folder
    for item in os.listdir(root_folder):
        item_path = os.path.join(root_folder, item)
        
        # Check if the item is a directory (e.g., 'school', 'sports')
        if os.path.isdir(item_path):
            print(f"Processing category: {item}...")
            category_data[item] = count_word_frequencies(item_path, junk)
            
    return category_data

# --- Execution ---
root_folder = "emails"
stop = "junk.txt"

# Run the master function
all_stats = process_all_categories(root_folder, stop)

# Display results
print("\n--- Summary ---")
for category, counts in all_stats.items():
    print(f"Category: {category.upper()} | Unique words: {len(counts)}")
    print(f"Top 3 words: {counts.most_common(3)}")
    print("-" * 20)
