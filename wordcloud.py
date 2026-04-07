import os
import re
import nltk
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize

# Ensure resources are available
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

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

def process_text_to_lemmas(raw_text, lemmatizer, junk):
    """
    Converts a raw string into a list of cleaned, lemmatized words.
    """
    # 1. Tokenize the full sentence to preserve context for the tagger
    tokens = word_tokenize(raw_text.lower())
    
    # 2. Get POS tags for the whole sentence
    tagged_tokens = pos_tag(tokens)
    
    # 3. Lemmatize, then filter
    final_lemmas = []
    for word, tag in tagged_tokens:
        # Only process if it's alphanumeric
        if word.isalnum():
            w_pos = get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word, w_pos)
            
            # 4. Now check if the normalized 'lemma' is in our junk list
            if lemma not in junk and len(lemma) > 1:
                final_lemmas.append(lemma)
                
    return final_lemmas

def count_word_frequencies(folder_path, junk, lemmatizer):
    cnt = Counter()
    
    if not os.path.exists(folder_path):
        return cnt

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                # Process the file content
                content = file.read()
                lemmatized_sentence = process_text_to_lemmas(content, lemmatizer, junk)
                
                # Update counter with the list of lemmas
                cnt.update(lemmatized_sentence)
    return cnt

def process_all_categories(root_folder, stop_file):
    # Load assets once
    if os.path.exists(stop_file):
        with open(stop_file, 'r') as f:
            junk = {line.strip().lower() for line in f if line.strip()}
    else:
        junk = set()
        
    lemmatizer = WordNetLemmatizer()
    all_categories = {}

    # Traverse subdirectories
    for category_name in os.listdir(root_folder):
        category_path = os.path.join(root_folder, category_name)
        
        if os.path.isdir(category_path):
            print(f"Analyzing {category_name}...")
            all_categories[category_name] = count_word_frequencies(category_path, junk, lemmatizer)
            
    return all_categories

# --- RUN ---
results = process_all_categories("emails", "junk.txt")

for cat, count_obj in results.items():
    print(f"\nTop words in {cat.upper()}:")
    print(count_obj.most_common(5))
    
