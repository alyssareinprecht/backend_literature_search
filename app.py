from flask import Flask, request, jsonify, render_template
import pandas as pd
from ast import literal_eval

app = Flask(__name__, static_folder='static', template_folder='templates')

##### Load data in 
df = pd.read_csv('data/papers.csv')
df['keywords_scaled_importance'] = df['keywords_scaled_importance'].apply(literal_eval)
df['word_frequency_dict'] = df['word_frequency_dict'].apply(literal_eval)

##### Function to rank papers 
def rank_papers(df, keyword_list):
    keyword_list = keyword_list[:4]
    
    def score_paper(keywords_importance):
        keywords_dict = dict(keywords_importance)
        score = []
        for keyword in keyword_list:
            score.append(keywords_dict.get(keyword, 0))
        return tuple(score)
    
    df['score'] = df['keywords_scaled_importance'].apply(score_paper)
    sorted_df = df.sort_values(by='score', ascending=False, kind='mergesort')
    return sorted_df

@app.route('/')
def index():
    return render_template('index.html')

##### Handle requests 
@app.route('/rank_papers', methods=['POST'])
def get_ranked_papers():
    data = request.json
    keywords = data.get('keywords', [])
    ranked_df = rank_papers(df, keywords)
    papers = ranked_df[['title', 'word_frequency_dict']].to_dict(orient='records')
    return jsonify(papers)

if __name__ == '__main__':
    app.run(debug=True)
