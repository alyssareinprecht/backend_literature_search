import pandas as pd
import numpy as np
path = #path to original data 
df = pd.read_csv(path)

##### Preprocess Data for keyword extraction
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization and remove stop words
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Join tokens back to string
    return ' '.join(tokens)

df['processed_text'] = df['paper_text'].apply(preprocess_text)

##### predefine keywords of interest
keywords = ["artificial intelligence", "machine learning", "neural networks", "deep learning", "reinforcement learning",
            "supervised learning", "unsupervised learning", "data mining", "natural language processing", "computer vision",
            "decision trees", "random forests", "gradient boosting", "support vector machines", "clustering",
            "regression", "classification", "feature engineering", "dimensionality reduction", "optimization",
            "backpropagation", "convolutional neural networks", "recurrent neural networks", "long short-term memory",
            "generative adversarial networks", "reinforcement learning", "transfer learning", "multi-task learning",
            "federated learning", "anomaly detection", "text mining", "sentiment analysis", "word embeddings",
            "transformers", "BERT", "GPT", "tokenization", "sequence models", "speech recognition", "image recognition",
            "object detection", "image segmentation", "facial recognition", "optical character recognition", "autonomous vehicles",
            "robotics", "drones", "path planning", "SLAM", "human-robot interaction", "bioinformatics", "genomics",
            "health informatics", "electronic health records", "predictive modeling", "epidemiology", "finance",
            "algorithmic trading", "credit scoring", "fraud detection", "recommendation systems", "e-commerce",
            "customer segmentation", "social network analysis", "graph algorithms", "game theory", "multi-agent systems",
            "simulations", "Monte Carlo methods", "quantum computing", "quantum machine learning", "differential privacy",
            "secure multi-party computation", "blockchain", "smart contracts", "edge computing", "internet of things",
            "big data", "data lakes", "scalability", "high performance computing", "cloud computing", "mobile computing",
            "software engineering", "agile methodologies", "DevOps", "microservices", "APIs", "user experience",
            "human-computer interaction", "visualization", "augmented reality", "virtual reality", "digital twins",
            "sustainability", "renewable energy", "smart grids", "climate change", "ecology", "agriculture",
            "precision farming", "urban planning", "smart cities", "transportation systems", "traffic management",
            "supply chain management", "logistics", "manufacturing", "3D printing", "materials science", "bayes"]



##### Extract Keywords 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_individual_keyword_similarity(papers, keywords):
    vectorizer = TfidfVectorizer()
    document_matrix = vectorizer.fit_transform(papers['paper_text'])
    
    results_data = {}
    for keyword in keywords:
        keyword_vector = vectorizer.transform([keyword])
        similarities = cosine_similarity(document_matrix, keyword_vector).flatten()
        
        if similarities.max() > 0:
            normalized_scores = similarities / similarities.max()
        else:
            normalized_scores = np.zeros_like(similarities)
        
        results_data[keyword] = normalized_scores

    results_df = pd.DataFrame(results_data, index=papers.index)

    def normalize_scores(scores):
        if np.isclose(scores.sum(), 0):
            return [1.0 / len(scores)] * len(scores)
        else:
            normalized_scores = scores / scores.sum()
            rounded_scores = np.round(normalized_scores, 2)
            correction = 1 - rounded_scores.sum()
            index = np.argmax(rounded_scores) if correction > 0 else np.argmin(rounded_scores)
            rounded_scores[index] += correction
            return rounded_scores.tolist()

    def process_row(row):
        top_keywords = sorted(row.items(), key=lambda x: x[1], reverse=True)[:5]
        scores = np.array([score for _, score in top_keywords])
        normalized_scores = normalize_scores(scores)
        return [(keyword, score) for (keyword, _), score in zip(top_keywords, normalized_scores)]

    papers['keyword_scaled_importance'] = results_df.apply(process_row, axis=1)
    
    return papers

### Apply 
df_keywords = extract_individual_keyword_similarity(df, keywords)

### Remove unnecesary columns
df_keywords = df_keywords.drop(columns=['abstract', 'paper_text', 'event_type', 'pdf_name',
                                        'processed_text', 'year'])

##### Load libraries for word frequencies
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

##### Extract Word Frequencies 
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def process_text(text):
    # Define stopwords
    stop_words = set(stopwords.words('english'))
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Remove punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    
    # Remove stopwords, lowercase, filter out numbers and short words
    words = [word.lower() for word in words if word.lower() not in stop_words and not word.isdigit() and len(word) > 3]
    
    # Lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # Count word frequencies
    word_freq = Counter(words)
    
    return word_freq

### apply
word_cloud_data = []

for index, row in df.iterrows():
    # Extract id and paper text
    paper_id = row['id']
    paper_text = row['paper_text']
    
    # Process text to get word frequencies
    word_freq = process_text(str(paper_text))
    # Only look at most frequent 
    top_40_words = word_freq.most_common(40)

    # Append word cloud data to the list
    for word, frequency in top_40_words:
        word_cloud_data.append({'id': paper_id, 'word': word, 'frequency': frequency})

# Create DataFrame from word cloud data
word_cloud_df = pd.DataFrame(word_cloud_data)

##### Create Dictionaries 
def aggregate_keywords(group):
    return dict(zip(group['word'], group['frequency']))

result_df = word_cloud_df.groupby('id').apply(aggregate_keywords).reset_index(name='word_frequency_dict')

##### Save Dataframe
full_df = pd.merge(result_df, df_keywords, on='id', how='left')
full_df.to_csv('research_papers_with_wordinfo.csv', index=False)
