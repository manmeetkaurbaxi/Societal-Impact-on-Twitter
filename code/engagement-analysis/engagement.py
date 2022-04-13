import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats
from scipy.signal import savgol_filter
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import glob

SPAN = 150
WINDOW_LENGTH = 51
POLY_ORDER = 8

def calculateAverageEngagementsPerDay(dataframe):
    dataframe['engagement_rate'] = dataframe['like_count'].astype(int) + dataframe['reply_count'].astype(int) + dataframe['retweet_count'].astype(int) + dataframe['quote_count'].astype(int)
    
    engagements_per_day = dataframe.groupby(['created_at']).agg({'engagement_rate':'sum'}).reset_index()
    tweets_per_day = (dataframe.groupby(['created_at'])['tweet'].count()).to_frame('tweets_per_day')
    
    average_engagements_per_day = tweets_per_day.merge(engagements_per_day, how='inner', on='created_at')
    average_engagements_per_day['average_engagement_per_day'] = np.round((average_engagements_per_day['engagement_rate']/ (4 * average_engagements_per_day['tweets_per_day'])), 2)
    
    return average_engagements_per_day

HEALTH_ORGS_PATH = '../../data/health-organizations/with-sentiment-labels/'
LEADERS_PATH = '../../data/leaders/with-sentiment-labels/'

user_info_df = pd.read_csv('../../data/user_info_updated.csv')

leaders_avg_engagements_per_day_df = pd.DataFrame()

for file in glob.glob(LEADERS_PATH+'/*.csv'):
    user_df = pd.read_csv(file)
    username = user_df['username'].unique()[0]
    user_impact = user_info_df[user_info_df['username'] == username]['user_impact'].unique()[0]
    
    # Calculate average engagement per day & it's Exponential Moving Average
    user_avg_engagements_per_day = calculateAverageEngagementsPerDay(user_df)
    user_avg_engagements_per_day['EMA']= user_avg_engagements_per_day.iloc[:,3].ewm(span=SPAN, adjust=False).mean()
    user_avg_engagements_per_day['user'] = username  
    user_avg_engagements_per_day['user_impact'] = user_impact
    
    #  Calculate z-score & Remove outliers
    user_avg_engagements_per_day['zscore'] = stats.zscore(user_avg_engagements_per_day['EMA'])
    user_avg_engagements_per_day = user_avg_engagements_per_day[(user_avg_engagements_per_day.zscore >= -3) & (user_avg_engagements_per_day.zscore <= 3)]

    # Curve Smoothing
    user_avg_engagements_per_day['EMA:Degree8'] = savgol_filter(user_avg_engagements_per_day['EMA'], WINDOW_LENGTH, POLY_ORDER)
    
    # Add user-impact to EMA    
    user_avg_engagements_per_day['EMA*user_impact'] = user_avg_engagements_per_day['EMA:Degree8'].mul(user_avg_engagements_per_day['user_impact_new'])
    
    print(username+':', round(user_avg_engagements_per_day['EMA*user_impact'].mean(),3))

health_orgs_avg_engagements_per_day_df = pd.DataFrame()

for file in glob.glob(HEALTH_ORGS_PATH+'/*.csv'):
    user_df = pd.read_csv(file)
    username = user_df['username'].unique()[0]
    user_impact = user_info_df[user_info_df['username'] == username]['user_impact'].unique()[0]
    
    # Calculate average engagement per day & it's Exponential Moving Average
    user_avg_engagements_per_day = calculateAverageEngagementsPerDay(user_df)
    user_avg_engagements_per_day['EMA']= user_avg_engagements_per_day.iloc[:,3].ewm(span=SPAN, adjust=False).mean()
    user_avg_engagements_per_day['user'] = username    
    user_avg_engagements_per_day['user_impact'] = user_impact
    
    #  Calculate z-score & Remove outliers
    user_avg_engagements_per_day['zscore'] = stats.zscore(user_avg_engagements_per_day['EMA'])
    user_avg_engagements_per_day = user_avg_engagements_per_day[(user_avg_engagements_per_day.zscore >= -3) & (user_avg_engagements_per_day.zscore <= 3)]

    # Curve Smoothing
    user_avg_engagements_per_day['EMA:Degree8'] = savgol_filter(user_avg_engagements_per_day['EMA'], WINDOW_LENGTH, POLY_ORDER)
    
    # Add user-impact to EMA    
    user_avg_engagements_per_day['EMA*user_impact'] = user_avg_engagements_per_day['EMA:Degree8'].mul(user_avg_engagements_per_day['user_impact_new'])
    
    print(username+':', round(user_avg_engagements_per_day['EMA*user_impact'].mean(),3))
    
    # Combine all topics
    health_orgs_avg_engagements_per_day_df = health_orgs_avg_engagements_per_day_df.append(user_avg_engagements_per_day, ignore_index=True, sort=False)