from flask import Flask, g, jsonify, render_template
import pandas as pd
from ast import literal_eval

app = Flask (__name__)

##### Load Dataframe
df = pd.read_csv('data/research_papers_with_wordinfo.csv')
df['keywords_scaled_importance'] = df['keywords_scaled_importance'].apply(literal_eval)
df['word_frequency_dict'] = df['word_frequency_dict'].apply(literal_eval)

##### Function to retrieve papers with keywords 
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
