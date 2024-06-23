from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import pandas as pd
from ast import literal_eval
from wordcloud import WordCloud
import io
import base64

app = Flask(__name__, static_folder='static', template_folder='templates')

# Allow CORS for all domains on all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Load data
df = pd.read_csv('data/research_papers_with_wordinfo.csv')
df['keyword_scaled_importance'] = df['keyword_scaled_importance'].apply(literal_eval)
df['word_frequency_dict'] = df['word_frequency_dict'].apply(literal_eval)

# Function to rank papers
def rank_papers(df, include_keywords, exclude_keywords):
    include_keywords = include_keywords[:5]  # Limiting to 5

    def score_paper(keywords_importance):
        keywords_dict = dict(keywords_importance)
        score = sum(keywords_dict.get(keyword['keyword'], 0) * keyword['priority'] for keyword in include_keywords)
        return score if score > 0 else None

    if include_keywords:
        df['score'] = df['keyword_scaled_importance'].apply(score_paper)
        df = df[df['score'].notnull()]

    # Filter out papers that contain any of the excluded keywords
    def contains_excluded_keywords(keywords_importance):
        keywords_dict = dict(keywords_importance)
        return any(keyword['keyword'] in keywords_dict for keyword in exclude_keywords)

    if exclude_keywords:
        df = df[~df['keyword_scaled_importance'].apply(contains_excluded_keywords)]

    if include_keywords:
        sorted_df = df.sort_values(by='score', ascending=False, kind='mergesort')
    else:
        sorted_df = df

    return sorted_df

# Function to generate word cloud
def generate_word_cloud(word_freq_dict):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_dict)
    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/rank_papers', methods=['POST'])
def get_ranked_papers():
    try:
        data = request.json
        include_keywords = data.get('include', [])
        exclude_keywords = data.get('exclude', [])
        if not include_keywords and not exclude_keywords:
            papers = df[['title', 'word_frequency_dict', 'keyword_scaled_importance']].to_dict(orient='records')
        else:
            ranked_df = rank_papers(df, include_keywords, exclude_keywords)
            papers = ranked_df[['title', 'word_frequency_dict', 'keyword_scaled_importance']].to_dict(orient='records')
        return jsonify(papers)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/keywords', methods=['GET'])
def get_keywords():
    try:
        keywords = set()
        for keywords_importance in df['keyword_scaled_importance']:
            for keyword, _ in keywords_importance:
                keywords.add(keyword)
        return jsonify(list(keywords))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/all_papers', methods=['GET'])
def get_all_papers():
    try:
        papers = df[['title', 'word_frequency_dict']].to_dict(orient='records')
        return jsonify(papers)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_word_cloud', methods=['POST'])
def generate_word_cloud_endpoint():
    try:
        data = request.json
        word_freq_dict = data.get('word_frequency_dict', {})
        word_cloud_img = generate_word_cloud(word_freq_dict)
        return jsonify({'word_cloud_image': word_cloud_img})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
