from flask import Flask, request, jsonify, render_template
import pandas as pd
from ast import literal_eval

app = Flask(__name__, static_folder='static', template_folder='templates')

##### Load data in 
df = pd.read_csv('data/papers.csv')
df['keywords_scaled_importance'] = df['keywords_scaled_importance'].apply(literal_eval)
df['word_frequency_dict'] = df['word_frequency_dict'].apply(literal_eval)

##### Function to rank papers 
def rank_papers(df, include_keywords, exclude_keywords):
    include_keywords = include_keywords[:5]  # Limiting to 5

    def score_paper(keywords_importance):
        keywords_dict = dict(keywords_importance)
        # Check for exclusion
        if any(keyword in keywords_dict for keyword in exclude_keywords):
            return None
        # Calculate score for inclusion
        score = []
        for keyword in include_keywords:
            score.append(keywords_dict.get(keyword, 0))
        return tuple(score)
        
    df['score'] = df['keywords_scaled_importance'].apply(score_paper)
    df = df[df['score'] != None]
    sorted_df = df.sort_values(by='score', ascending=False, kind='mergesort')
    return sorted_df

@app.route('/')
def index():
    return render_template('index.html')

##### Handle requests 
@app.route('/rank_papers', methods=['POST'])
def get_ranked_papers():
    data = request.json
    include_keywords = data.get('include', [])
    exclude_keywords = data.get('exclude', [])
    ranked_df = rank_papers(df, include_keywords, exclude_keywords)
    papers = ranked_df[['title', 'word_frequency_dict']].to_dict(orient='records')
    return jsonify(papers)

if __name__ == '__main__':
    app.run(debug=True)
