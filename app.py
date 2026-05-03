# ============================================
# FILE 4: app.py
# PURPOSE: Flask API for KyaKre career
#          recommendation engine
# ============================================

from flask import Flask, request, jsonify, render_template
from recommender import setup, get_recommendations

app = Flask(__name__)

print("Starting KyaKre server...")
df, tfidf, tfidf_matrix, similarity_matrix = setup()
print("Server ready!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    if not data or 'product' not in data:
        return jsonify({"error": "Please provide a career name"}), 400
    career_name = data['product']
    if not career_name.strip():
        return jsonify({"error": "Career name cannot be empty"}), 400
    result = get_recommendations(career_name, df, similarity_matrix)
    return jsonify(result)

@app.route('/careers', methods=['GET'])
def get_careers():
    stream = request.args.get('stream', None)
    if stream:
        filtered = df[df['stream'].str.lower() == stream.lower()]
    else:
        filtered = df
    careers = filtered[['id', 'name', 'stream', 'category', 'avg_salary_lpa', 'rating']].to_dict(orient='records')
    return jsonify({"total": len(careers), "careers": careers})

@app.route('/streams', methods=['GET'])
def get_streams():
    streams = df.groupby('stream')['name'].count().to_dict()
    return jsonify(streams)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "running",
        "total_careers": len(df),
        "message": "KyaKre NLP Career Recommendation Engine is live!"
    })

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  KyaKre Career Recommendation Engine")
    print("  Open browser: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True)