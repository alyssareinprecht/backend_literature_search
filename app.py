from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from ast import literal_eval

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Load data
df = pd.read_csv('data/research_papers_with_wordinfo.csv')
df['keyword_scaled_importance'] = df['keyword_scaled_importance'].apply(literal_eval)
df['word_frequency_dict'] = df['word_frequency_dict'].apply(literal_eval)

# Function to rank papers
def rank_papers(df, include_keywords, exclude_keywords):
    include_keywords = include_keywords[:5]  # Limiting to 5

    def score_paper(keywords_importance):
        keywords_dict = dict(keywords_importance)
        score = sum(keywords_dict.get(keyword, 0) for keyword in include_keywords)
        return score if score > 0 else None

    df['score'] = df['keyword_scaled_importance'].apply(score_paper)
    df = df[df['score'].notnull()]
    sorted_df = df.sort_values(by='score', ascending=False, kind='mergesort')
    return sorted_df

@app.route('/')
def index():
    return render_template('index.html')

# Handle requests
@app.route('/rank_papers', methods=['POST'])
def get_ranked_papers():
    data = request.json
    include_keywords = data.get('include', [])
    exclude_keywords = data.get('exclude', [])

    # Check if both include and exclude keywords are empty
    if not include_keywords and not exclude_keywords:
        papers = df[['title', 'word_frequency_dict']].to_dict(orient='records')
    else:
        ranked_df = rank_papers(df, include_keywords, exclude_keywords)
        papers = ranked_df[['title', 'word_frequency_dict']].to_dict(orient='records')
    
    return jsonify(papers)

@app.route('/keywords', methods=['GET'])
def get_keywords():
    keywords = set()
    for keywords_importance in df['keyword_scaled_importance']:
        for keyword, _ in keywords_importance:
            keywords.add(keyword)
    return jsonify(list(keywords))

# Add the new endpoint to fetch all papers
@app.route('/all_papers', methods=['GET'])
def get_all_papers():
    papers = df[['title', 'word_frequency_dict']].to_dict(orient='records')
    return jsonify(papers)

if __name__ == '__main__':
    app.run(debug=True)
