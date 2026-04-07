'''
This file will be used to sort through existing/given email data
by getting the position of a word, its semantic meaning, lemmatizing
text, the likes
'''
import os
from collections import Counter
import nltk 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet 
from nltk import pos_tag, word_tokenize 

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def load_junk_words(filepath):
    """Reads the stopword file and returns a set of words."""
    junk = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            junk = {line.strip().lower() for line in f if line.strip()}
    return junk

def lemma(text, lemmatizer, junk):
    """Tokenizes, tags, and lemmatizes a raw string of text."""
    # word_tokenize expects a string, not a list!
    tokens = word_tokenize(text.lower())
    tagged_tokens = pos_tag(tokens)
    
    final_lemmas = []
    for word, tag in tagged_tokens:
        if word.isalnum():
            w_pos = get_wordnet_pos(tag)
            lem = lemmatizer.lemmatize(word, w_pos)
            
            # Check if the normalized 'lem' is in our junk list
            if lem not in junk and len(lem) > 1:
                final_lemmas.append(lem)
                
    return final_lemmas
    
def count_word_frequencies(folder_path, junk, lemmatizer):
    """Counts frequencies for a single specific folder."""
    cnt = Counter()
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                # Read the WHOLE file as one string for better NLP context
                text = file.read()
                filt = lemma(text, lemmatizer, junk)
                cnt.update(filt)
    return cnt  
    
def handle(email_dir, lemmatizer, junk_set):
    """Iterates through category folders and returns a dictionary of counters."""
    category_data = {}
    
    if not os.path.exists(email_dir):
        print(f"Error: Directory '{email_dir}' not found.")
        return category_data
        
    for item in os.listdir(email_dir):
        item_path = os.path.join(email_dir, item)
        
        if os.path.isdir(item_path):
            print(f"Processing category: {item}...")
            category_data[item] = count_word_frequencies(item_path, junk_set, lemmatizer)
                
    return category_data

root_folder = "emails"
stop = load_junk_words("junk.txt")
wnl = WordNetLemmatizer()
stats = handle(root_folder, wnl, stop)

print("\n--- Summary ---")
for category, counts in stats.items():
    print(f"Category: {category.upper()} | Unique words: {len(counts)}")
    print(f"Top 5 words: {counts.most_common(5)}")
    print("-" * 20)
