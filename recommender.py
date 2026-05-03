# ============================================
# FILE 3: recommender.py
# PURPOSE: Career recommendation engine
#          for KyaKre website
# ============================================

import pandas as pd
from data_preprocessing import load_data, clean_data, create_features
from nlp_engine import build_tfidf_matrix, build_similarity_matrix

def setup():
    print("Setting up KyaKre recommendation engine...")
    df = load_data()
    df = clean_data(df)
    df = create_features(df)
    tfidf, tfidf_matrix = build_tfidf_matrix(df)
    similarity_matrix = build_similarity_matrix(tfidf_matrix)
    print("✅ Engine ready!\n")
    return df, tfidf, tfidf_matrix, similarity_matrix

def find_career(df, career_name):
    career_name = career_name.lower().strip()
    
    # First try exact partial match in name
    matches = df[df['name'].str.lower().str.contains(career_name)]
    if not matches.empty:
        return matches.index[0], df['name'][matches.index[0]]

    # Try category
    matches = df[df['category'].str.lower().str.contains(career_name)]
    if not matches.empty:
        return matches.index[0], df['name'][matches.index[0]]

    # Try stream
    matches = df[df['stream'].str.lower().str.contains(career_name)]
    if not matches.empty:
        return matches.index[0], df['name'][matches.index[0]]

    # Try skills
    matches = df[df['skills_needed'].str.lower().str.contains(career_name)]
    if not matches.empty:
        return matches.index[0], df['name'][matches.index[0]]

    # Try description
    matches = df[df['description'].str.lower().str.contains(career_name)]
    if not matches.empty:
        return matches.index[0], df['name'][matches.index[0]]

    # FUZZY MATCH — handles spelling mistakes like "dictor" → "doctor"
    from fuzzywuzzy import process
    all_names = df['name'].tolist()
    best_match, score = process.extractOne(career_name, all_names)
    
    if score >= 50:  # 50% similarity is enough
        idx = df[df['name'] == best_match].index[0]
        print(f"Fuzzy matched '{career_name}' → '{best_match}' (score: {score})")
        return idx, best_match
    
    return None, f"No career found matching '{career_name}'. Try: doctor, engineer, lawyer, designer, CA, pilot"

def get_recommendations(career_name, df, similarity_matrix, top_n=5):
    idx, matched_name = find_career(df, career_name)

    if idx is None:
        return {"error": matched_name}

    print(f"Finding recommendations for: {matched_name}")

    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:]
    top_careers = similarity_scores[:top_n]

    recommendations = []
    for career_idx, score in top_careers:
        recommendations.append({
            "name": df['name'][career_idx],
            "stream": df['stream'][career_idx],
            "category": df['category'][career_idx],
            "rating": df['rating'][career_idx],
            "avg_salary_lpa": int(df['avg_salary_lpa'][career_idx]),
            "skills_needed": df['skills_needed'][career_idx],
            "top_colleges": df['top_colleges'][career_idx],
            "similarity_score": round(score, 4),
            "description": df['description'][career_idx]
        })

    return {
        "searched_career": matched_name,
        "stream": df['stream'][idx],
        "category": df['category'][idx],
        "recommendations": recommendations
    }

if __name__ == "__main__":
    df, tfidf, tfidf_matrix, similarity_matrix = setup()

    print("=" * 60)
    print("TEST 1: Searching for 'doctor'")
    print("=" * 60)
    result = get_recommendations("doctor", df, similarity_matrix)
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"{i}. {rec['name']} ({rec['stream']})")
        print(f"   Category : {rec['category']}")
        print(f"   Salary   : {rec['avg_salary_lpa']} LPA")
        print(f"   Similarity: {rec['similarity_score']}")
        print()

    print("=" * 60)
    print("TEST 2: Searching for 'engineer'")
    print("=" * 60)
    result2 = get_recommendations("engineer", df, similarity_matrix)
    for i, rec in enumerate(result2['recommendations'], 1):
        print(f"{i}. {rec['name']} ({rec['stream']})")
        print(f"   Category : {rec['category']}")
        print(f"   Salary   : {rec['avg_salary_lpa']} LPA")
        print(f"   Similarity: {rec['similarity_score']}")
        print()

    print("✅ KyaKre Recommender working!")