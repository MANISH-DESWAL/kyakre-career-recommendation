# ============================================
# FILE 2: nlp_engine.py
# PURPOSE: Convert text to numbers using
#          TF-IDF and calculate similarity
# ============================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_preprocessing import load_data, clean_data, create_features

# ----------------------------
# STEP 1: Build TF-IDF Matrix
# ----------------------------
def build_tfidf_matrix(df):
    """
    TF-IDF converts each product's text into a row of numbers.
    Each number represents how important a word is for that product.
    
    Example:
    'wireless battery bluetooth' becomes --> [0.45, 0.32, 0.67, 0.0, ...]
    'bass foldable design'       becomes --> [0.0,  0.0,  0.12, 0.54, ...]
    """
    
    # Create TF-IDF Vectorizer
    # max_features=5000 means we only track the 5000 most important words
    tfidf = TfidfVectorizer(max_features=5000)
    
    # Fit and transform - this builds the matrix
    # Each row = one product
    # Each column = one word
    # Each value = importance score of that word for that product
    tfidf_matrix = tfidf.fit_transform(df['combined_text'])
    
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    print(f"→ {tfidf_matrix.shape[0]} products x {tfidf_matrix.shape[1]} unique words")
    
    return tfidf, tfidf_matrix

# ----------------------------
# STEP 2: Calculate Similarity
# ----------------------------
def build_similarity_matrix(tfidf_matrix):
    """
    Cosine similarity measures the angle between two product vectors.
    
    Score = 1.0 → products are identical
    Score = 0.5 → products are somewhat similar  
    Score = 0.0 → products are completely different
    
    We calculate similarity between ALL pairs of products at once.
    Result is a 30x30 matrix where each cell = similarity score.
    """
    
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    print(f"\nSimilarity Matrix Shape: {similarity_matrix.shape}")
    print(f"→ {similarity_matrix.shape[0]}x{similarity_matrix.shape[1]} = every product vs every product")
    
    return similarity_matrix

# ----------------------------
# STEP 3: Show sample scores
# ----------------------------
def show_sample_similarities(df, similarity_matrix):
    """
    Let's peek at some similarity scores to verify it's working
    """
    print("\n--- Sample Similarity Scores for Product 0 (boAt Headphone) ---")
    
    scores = list(enumerate(similarity_matrix[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Product Name':<45} {'Score':>8}")
    print("-" * 55)
    for idx, score in scores[:6]:
        print(f"{df['name'][idx]:<45} {score:>8.4f}")

# ----------------------------
# MAIN: Run all steps
# ----------------------------
if __name__ == "__main__":
    # Load and preprocess data
    df = load_data()
    df = clean_data(df)
    df = create_features(df)
    
    print("\n--- Building NLP Engine ---")
    
    # Build TF-IDF matrix
    tfidf, tfidf_matrix = build_tfidf_matrix(df)
    
    # Build similarity matrix
    similarity_matrix = build_similarity_matrix(tfidf_matrix)
    
    # Show sample results
    show_sample_similarities(df, similarity_matrix)
    
    print("\n✅ NLP Engine ready!")
    