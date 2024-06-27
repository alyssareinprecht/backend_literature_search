from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import pandas as pd
from ast import literal_eval
from wordcloud import WordCloud
import io
import base64

# Set up Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')

# Allows CORS (Cross Origin Resource Sharing) for all domains on all routes
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the data of documents
df = pd.read_csv('data/research_papers_with_wordinfo.csv')
df['keyword_scaled_importance'] = df['keyword_scaled_importance'].apply(literal_eval)
df['word_frequency_dict'] = df['word_frequency_dict'].apply(literal_eval)

# Function to rank documents based on included and excluded keywords from the user
def rank_papers(df, include_keywords, exclude_keywords):
    include_keywords = include_keywords[:5]  # Limiting to 5

    # Function to score documents based on keyword importance and priority selected by user
    def score_paper(keywords_importance):
        keywords_dict = dict(keywords_importance)
        score = sum(keywords_dict.get(keyword['keyword'], 0) * keyword['priority'] for keyword in include_keywords)
        return score if score > 0 else None

    # Filters documents if keywords are included
    if include_keywords:
        df['score'] = df['keyword_scaled_importance'].apply(score_paper)
        df = df[df['score'].notnull()]

    # Function to check if excluded keywords are selected
    def contains_excluded_keywords(keywords_importance):
        keywords_dict = dict(keywords_importance)
        return any(keyword['keyword'] in keywords_dict for keyword in exclude_keywords)

    # Filters out papers that contain excluded keywords
    if exclude_keywords:
        df = df[~df['keyword_scaled_importance'].apply(contains_excluded_keywords)]

    # Sorts documents by score if included keywords are selected
    if include_keywords:
        sorted_df = df.sort_values(by='score', ascending=False, kind='mergesort')
    else:
        sorted_df = df

    return sorted_df

# Function to generate word clouds from the word frequency dictionary
def generate_word_cloud(word_freq_dict):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq_dict)
    img = io.BytesIO()
    wordcloud.to_image().save(img, format='PNG')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# Route to serve index.html 
@app.route('/')
def index():
    return render_template('index.html')

# Route to the rank documents functionality based on included and excluded keywords
@app.route('/rank_papers', methods=['POST'])
def get_ranked_papers():
    try:
        data = request.json
        include_keywords = data.get('include', [])
        exclude_keywords = data.get('exclude', [])

        # Check if the number of exclude keywords exceeds the limit
        if len(exclude_keywords) > 15:
            return jsonify({'error': 'You can only exclude up to 15 keyword tags.'}), 400

        # Returns all paper if no keywords are selected
        if not include_keywords and not exclude_keywords:
            papers = df[['title', 'word_frequency_dict', 'keyword_scaled_importance']].to_dict(orient='records')
        else:
            ranked_df = rank_papers(df, include_keywords, exclude_keywords)
            papers = ranked_df[['title', 'word_frequency_dict', 'keyword_scaled_importance']].to_dict(orient='records')
        return jsonify(papers)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to return all keywords
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

# Route to return all documents with word frequency
@app.route('/all_papers', methods=['GET'])
def get_all_papers():
    try:
        papers = df[['title', 'word_frequency_dict']].to_dict(orient='records')
        return jsonify(papers)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Route to create word cloud images 
@app.route('/generate_word_cloud', methods=['POST'])
def generate_word_cloud_endpoint():
    try:
        data = request.json
        word_freq_dict = data.get('word_frequency_dict', {})
        word_cloud_img = generate_word_cloud(word_freq_dict)
        return jsonify({'word_cloud_image': word_cloud_img})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
