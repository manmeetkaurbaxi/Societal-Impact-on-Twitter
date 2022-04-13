import pandas as pd
import tweepy
import math
import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, minmax_scale
import plotly.express as px

bearer_token = ""
client = tweepy.Client(bearer_token=bearer_token)

start_time = '2019-12-01T00:00:00Z'
end_time = '2021-12-31T23:59:59Z'

usernames = ['sebastianpinera','PresidentIRL','MichealMartinTD','HHShkMohd','MarinSanna','JustinTrudeau','CanadianPM','IvanDuque','RTErdogan','sanchezcastejon','SwedishPM','BorisJohnson',
              'ministeriosalud','HSELive','roinnslainte','mohapuae','MSAH_News','GovCanHealth','MinSaludCol','saglikbakanligi','sanidadgob','Folkhalsomynd','UKHSA']

user_info = client.get_users(usernames=usernames, user_fields=['created_at','public_metrics','description','location','verified'])

user_info_df = pd.DataFrame(columns=['created_at','name','username','followers_count','following_count','tweet_count','listed_count','description','location','verified'])

for user in user_info.data:    
    user_info_df = user_info_df.append({'created_at':user.created_at, 'name':user.name, 'username':user.username, 'followers_count':user.public_metrics.get('followers_count'),
                                        'following_count': user.public_metrics.get('following_count'), 'tweet_count':user.public_metrics.get('tweet_count'), 
                                        'listed_count':user.public_metrics.get('listed_count'), 'description':user.description, 'location':user.location, 'verified':user.verified}, 
                                       ignore_index=True)

user_info_df.to_csv('../../data/user_info.csv', index=False)