# ============================================
# FILE 1: data_preprocessing.py
# PURPOSE: Load and clean career dataset
#          for KyaKre recommendation engine
# ============================================

import pandas as pd
import nltk
import re

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from nltk.corpus import stopwords

def load_data():
    df = pd.read_csv('data/products.csv')
    print(f"Dataset loaded: {df.shape[0]} careers, {df.shape[1]} columns")
    return df

def clean_data(df):
    df = df.dropna(subset=['name', 'description'])
    df = df.drop_duplicates(subset=['name'])
    df = df.reset_index(drop=True)
    print(f"After cleaning: {df.shape[0]} careers remaining")
    return df

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def create_features(df):
    # Combine ALL relevant columns for better NLP matching
    df['combined_text'] = (
        df['name'] + ' ' +
        df['stream'] + ' ' +
        df['category'] + ' ' +
        df['description'] + ' ' +
        df['skills_needed']
    )
    df['combined_text'] = df['combined_text'].apply(clean_text)
    print("Feature engineering done!")
    return df

if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = create_features(df)
    print("\n✅ Preprocessing complete!")
    print(df[['name', 'stream', 'combined_text']].head(3))