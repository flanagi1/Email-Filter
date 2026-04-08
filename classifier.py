'''
This file will be used to classify any new emails coming in based on 
specified categories given by the user/training data. Naive Bayes will 
eventually play a role in this
'''
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
    
'''LEMMA AND REMOVE STOP WORDS FROM NEW EMAIL, SOME PROCESS AS EARLIER 
ONCE WE HIT THAT, START THE CALCULATION

CAN SCALE TO USING GMAIL API LATER BUT START SMALL FIRST'''
