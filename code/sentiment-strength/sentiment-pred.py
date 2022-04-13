import pandas as pd
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import glob

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []

    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

sentiment_model_path = f"cardiffnlp/twitter-roberta-base-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)

# Download label mapping
sentiment_labels = []
sentiment_mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
with urllib.request.urlopen(sentiment_mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
sentiment_labels = [row[1] for row in csvreader if len(row) > 1]

sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    sentiment_model_path)
sentiment_model.save_pretrained(sentiment_model_path)
sentiment_tokenizer.save_pretrained(sentiment_model_path)

# Constants
HEALTH_ORGS_PATH = 'data/with-sentiment-labels/health-organizations/'
LEADERS_PATH = 'data/with-sentiment-labels/leaders/'

for file in glob.glob(LEADERS_PATH+'/*.csv'):
    user_df = pd.read_csv(file)
    username = user_df['username'].unique()[0]
    print(username)
    
    for index, row in user_df.iterrows():
        tweet = row['tweet_en']
        tweet = preprocess(tweet)

        # Calculate sentiment
        sentiment_encoded_input = sentiment_tokenizer(
            tweet, return_tensors='pt')
        sentiment_output = sentiment_model(**sentiment_encoded_input)
        sentiment_scores = sentiment_output[0][0].detach().numpy()
        sentiment_scores = softmax(sentiment_scores)

        sentiment_ranking = np.argsort(sentiment_scores)
        sentiment_ranking = sentiment_ranking[::-1]
        # print(sentiment_labels[sentiment_ranking[0]])
        for i in range(sentiment_scores.shape[0]):
            sentiment_label = sentiment_labels[sentiment_ranking[i]]
            sentiment_score = sentiment_scores[sentiment_ranking[i]]
            # print(f'{sentiment_label} {np.round(float(sentiment_score), 4)}')
            user_df.at[index, sentiment_label] = np.round(
                float(sentiment_score), 6)
        user_df.at[index, 'sentiment'] = sentiment_labels[sentiment_ranking[0]]
    
    user_df.to_csv('data/with-sentiment-labels/leaders/'+username+'.csv', index=False)
    
for file in glob.glob(HEALTH_ORGS_PATH+'/*.csv'):
    user_df = pd.read_csv(file)
    username = user_df['username'].unique()[0]
    print(username)
    
    for index, row in user_df.iterrows():
        tweet = row['tweet_en']
        tweet = preprocess(tweet)

        # Calculate sentiment
        sentiment_encoded_input = sentiment_tokenizer(
            tweet, return_tensors='pt')
        sentiment_output = sentiment_model(**sentiment_encoded_input)
        sentiment_scores = sentiment_output[0][0].detach().numpy()
        sentiment_scores = softmax(sentiment_scores)

        sentiment_ranking = np.argsort(sentiment_scores)
        sentiment_ranking = sentiment_ranking[::-1]
        # print(sentiment_labels[sentiment_ranking[0]])
        for i in range(sentiment_scores.shape[0]):
            sentiment_label = sentiment_labels[sentiment_ranking[i]]
            sentiment_score = sentiment_scores[sentiment_ranking[i]]
            # print(f'{sentiment_label} {np.round(float(sentiment_score), 4)}')
            user_df.at[index, sentiment_label] = np.round(
                float(sentiment_score), 6)
        user_df.at[index, 'sentiment'] = sentiment_labels[sentiment_ranking[0]]
    
    user_df.to_csv('data/with-sentiment-labels/health-organizations'+username+'.csv', index=False)