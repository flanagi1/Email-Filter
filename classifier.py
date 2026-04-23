'''
This file will be used to classify any new emails coming in based on 
specified categories given by the user/training data. Naive Bayes will 
 play a role in this
'''
import math
import json
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

import math

def classify_email(processed_words, trained_data):
    # 1. Basic Metadata
    categories = trained_data["word_counts"].keys()
    email_counts = trained_data["metadata"]["category_email_counts"]
    total_emails = trained_data["metadata"]["total_emails"]
    
    # 2. Calculate Global Vocabulary Size (Total unique words across all categories)
    all_unique_words = set()
    for cat in categories:
        all_unique_words.update(trained_data["word_counts"][cat].keys())
    vocab_size = len(all_unique_words)
    
    results = {}

    for cat in categories:
        # --- PHASE A: PRIOR PROBABILITY ---
        # P(Category) = emails_in_cat / total_emails
        prior = email_counts[cat] / total_emails
        # We start our score with the log of the prior
        score = math.log(prior)
        
        # --- PHASE B: LIKELIHOOD ---
        cat_word_data = trained_data["word_counts"][cat]
        # Sum of all word occurrences in this category
        total_word_count_in_cat = sum(cat_word_data.values())
        
        for word in processed_words:
            # Get count of this specific word in this category (default to 0)
            word_count = cat_word_data.get(word, 0)
            
            # Laplace Smoothing Formula: (count + 1) / (total + vocab_size)
            word_probability = (word_count + 1) / (total_word_count_in_cat + vocab_size)
            
            # Add the log of the probability to our score
            score += math.log(word_probability)
            
        results[cat] = score

    return results

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


with open("trained_data.json", "r") as f:
    trained_data = json.load(f)
    
junk = load_junk_words("junk.txt")
wnl = WordNetLemmatizer()

new_email_path = "new_email.txt"

if os.path.exists(new_email_path):
    with open(new_email_path, "r", encoding='utf-8') as mail:
        content = mail.read()
        
        # Use your lemma function directly to get the word list
        processed_words = lemma(content, wnl, junk)
        
        # If you want the frequency count of words in THIS specific email:
        new_email_counts = Counter(processed_words)

    print("\n--- New Email Top Words ---")
    print(new_email_counts.most_common(10))
else:
    print("Error: new_email.txt not found.")
  

# --- Running the Classifier ---
# 'processed_words' is the array of lemmas from your new email
final_scores = classify_email(processed_words, trained_data)

# Sort to find the winner
winner = max(final_scores, key=final_scores.get)

print("\n--- Naive Bayes Scores (Log Scale) ---")
for cat, score in final_scores.items():
    print(f"{cat.upper()}: {score:.4f}")

print(f"\nResult: This email is likely categorized as '{winner.upper()}'")
  
'''LEMMA AND REMOVE STOP WORDS FROM NEW EMAIL, SOME PROCESS AS EARLIER 
ONCE WE HIT THAT, START THE CALCULATION

CAN SCALE TO USING GMAIL API LATER BUT START SMALL FIRST'''
