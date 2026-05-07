'''
Naive Bayes email classifier.
Can be run as a script (classifies new_email.txt) OR imported as a module.
Usage as module:
    from classifier import classify_text
    winner, scores = classify_text("your email body here")
'''
import math
import json
import os
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize

# --- NLP Helpers ---

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
    """Tokenizes, POS-tags, and lemmatizes a raw string of text."""
    tokens = word_tokenize(text.lower())
    tagged_tokens = pos_tag(tokens)
    final_lemmas = []
    for word, tag in tagged_tokens:
        if word.isalnum():
            w_pos = get_wordnet_pos(tag)
            lem = lemmatizer.lemmatize(word, w_pos)
            if lem not in junk and len(lem) > 1:
                final_lemmas.append(lem)
    return final_lemmas


# --- Core Classifier ---

def classify_email(processed_words, trained_data):
    """
    Runs Naive Bayes on a list of pre-processed words.
    Returns a dict of {category: log_probability_score}.
    """
    categories = trained_data["word_counts"].keys()
    email_counts = trained_data["metadata"]["category_email_counts"]
    total_emails = trained_data["metadata"]["total_emails"]

    all_unique_words = set()
    for cat in categories:
        all_unique_words.update(trained_data["word_counts"][cat].keys())
    vocab_size = len(all_unique_words)

    results = {}
    for cat in categories:
        prior = email_counts[cat] / total_emails
        score = math.log(prior)

        cat_word_data = trained_data["word_counts"][cat]
        total_word_count_in_cat = sum(cat_word_data.values())

        for word in processed_words:
            word_count = cat_word_data.get(word, 0)
            word_probability = (word_count + 1) / (total_word_count_in_cat + vocab_size)
            score += math.log(word_probability)

        results[cat] = score

    return results


# --- Public API (for use by skills and other scripts) ---

# Resolve paths relative to this file so the module works from any working directory
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAINED_DATA_PATH = os.path.join(_THIS_DIR, "trained_data.json")
_JUNK_PATH = os.path.join(_THIS_DIR, "junk.txt")

# Lazy-loaded globals (initialized on first call to classify_text)
_trained_data = None
_junk = None
_wnl = None


def _init():
    """Load model assets once and cache them."""
    global _trained_data, _junk, _wnl
    if _trained_data is None:
        with open(_TRAINED_DATA_PATH, "r") as f:
            _trained_data = json.load(f)
        _junk = load_junk_words(_JUNK_PATH)
        _wnl = WordNetLemmatizer()


def classify_text(text: str):
    """
    Classify a raw email string.

    Returns:
        winner (str): The predicted category name.
        scores (dict): Log-probability scores for every category.
    """
    _init()
    processed_words = lemma(text, _wnl, _junk)
    scores = classify_email(processed_words, _trained_data)
    winner = max(scores, key=scores.get)
    return winner, scores


# --- Script mode (python classifier.py) ---

if __name__ == "__main__":
    new_email_path = os.path.join(_THIS_DIR, "new_email.txt")

    if not os.path.exists(new_email_path):
        print("Error: new_email.txt not found.")
    else:
        with open(new_email_path, "r", encoding="utf-8") as f:
            content = f.read()

        winner, scores = classify_text(content)

        print("\n--- Naive Bayes Scores (Log Scale) ---")
        for cat, score in scores.items():
            print(f"{cat.upper()}: {score:.4f}")
        print(f"\nResult: This email is likely categorized as '{winner.upper()}'")
