import numpy as np
import pandas as pd
import glob

HEALTH_ORGS_PATH = '../../data/with-sentiment-labels/health-organizations/'
LEADERS_PATH = '../../data/with-sentiment-labels/leaders/'

# For all leaders
for file in glob.glob(LEADERS_PATH+'/*.csv'):
    user_df = pd.read_csv(file)
    username = user_df['username'].unique()[0]
    totalTweets = user_df.shape[0]
    
    overall_sentiments = (user_df.groupby(['sentiment'])['tweet'].count()).to_frame('sentimentTweetCount')
    overall_sentiments.reset_index(level=['sentiment'], inplace=True)
    maxSentiment = overall_sentiments.iloc[overall_sentiments['sentimentTweetCount'].idxmax()][['sentiment']].values[0]
    
    # print(maxSentiment)
    if maxSentiment == 'neutral':
        sentimentStrength = 0.001
        print(username, sentimentStrength)
    elif maxSentiment == 'positive':
        positiveTweets = overall_sentiments[overall_sentiments['sentiment'] == 'positive'][['sentimentTweetCount']].values[0][0]
        sentimentStrength = np.round(positiveTweets/totalTweets,3)
        print(username, sentimentStrength)
    elif maxSentiment == 'negative':
        negativeTweets = overall_sentiments[overall_sentiments['sentiment'] == 'negative'][['sentimentTweetCount']].values[0][0]
        sentimentStrength = np.round(-1*(negativeTweets/totalTweets),3)
        print(username, sentimentStrength)
    
    print('*'*50)

# For all health organizations
for file in glob.glob(HEALTH_ORGS_PATH+'/*.csv'):
    user_df = pd.read_csv(file)
    username = user_df['username'].unique()[0]
    totalTweets = user_df.shape[0]
    
    overall_sentiments = (user_df.groupby(['sentiment'])['tweet'].count()).to_frame('sentimentTweetCount')
    overall_sentiments.reset_index(level=['sentiment'], inplace=True)
    maxSentiment = overall_sentiments.iloc[overall_sentiments['sentimentTweetCount'].idxmax()][['sentiment']].values[0]
    
    # print(maxSentiment)
    if maxSentiment == 'neutral':
        sentimentStrength = 0.001
        print(username, sentimentStrength)
    elif maxSentiment == 'positive':
        positiveTweets = overall_sentiments[overall_sentiments['sentiment'] == 'positive'][['sentimentTweetCount']].values[0][0]
        sentimentStrength = np.round(positiveTweets/totalTweets,3)
        print(username, sentimentStrength)
    elif maxSentiment == 'negative':
        negativeTweets = overall_sentiments[overall_sentiments['sentiment'] == 'negative'][['sentimentTweetCount']].values[0][0]
        sentimentStrength = np.round(-1*(negativeTweets/totalTweets),3)
        print(username, sentimentStrength)
    
    print('*'*50)