import pandas as pd
import numpy as np
import glob

# Lists as per different subsets
gender = ['male','female','transgender','lgbtq','men','women','girls','boys','fathers','mothers']
age = ['children','child','youngsters','adults','elders','youth','parents','grandparents','family','families']
culturalInferences = ['spanish','indigenious','gaelic','english','scottish','anglo-norman','arabian','islamic','persian','nordic','scandavian','british','french','african','anatolian','ottoman','celtic','pre-roman liberarian','european']
ethnicity = ['mesitzo','celts','irish','indian','pakistan','bangladesh','iran','finnish','swedish','sami','roma','scottish','russian','norwegian','norway','native american indian','east asian','turks','neolithic','visigoths','greek','romans','pheonici ans moors','scandinavian','syrian','iraqi','white']
employmentSectors = ['health care','construction','tourism','manufacturing','agriculture','energy','machinery','textile','electronics','mining','automobile','logging','petroleum','retail']

# Concat all subsets into one list
allCommunities = list(dict.fromkeys(gender + age + culturalInferences + ethnicity + employmentSectors))

# Check for community mentions
HEALTH_ORGS_PATH = '../../data/with-sentiment-labels/health-organizations/'
LEADERS_PATH = '../../data/with-sentiment-labels/leaders/'

# For all leaders
for file in glob.glob(LEADERS_PATH+'/*.csv'):
    user_df = pd.read_csv(file)
    username = user_df['username'].unique()[0]
    totalTweets = user_df.shape[0]
    
    communityMentions = 0
    for index, row in user_df.iterrows():
        for phrase in allCommunities:
            tweet_en = str(row['tweet_en'])
            if phrase in tweet_en:
                communityMentions += 1
                # print(row)
    print(username)                
    print('Community Mentions:',communityMentions)
    inclusivityDiversityRatio = np.round(communityMentions/totalTweets, 3)
    print('Inclusivity/Diversity Strength:', inclusivityDiversityRatio)
    print('*'*50)

# For all health organizations
for file in glob.glob(HEALTH_ORGS_PATH+'/*.csv'):
    user_df = pd.read_csv(file)
    username = user_df['username'].unique()[0]
    totalTweets = user_df.shape[0]
    
    communityMentions = 0
    for index, row in user_df.iterrows():
        for phrase in allCommunities:
            tweet_en = str(row['tweet_en'])
            if phrase in tweet_en:
                communityMentions += 1
                # print(row)
    print(username)                
    print('Community Mentions:',communityMentions)
    inclusivityDiversityRatio = np.round(communityMentions/totalTweets, 3)
    print('Inclusivity/Diversity Strength:', inclusivityDiversityRatio)
    print('*'*50)


