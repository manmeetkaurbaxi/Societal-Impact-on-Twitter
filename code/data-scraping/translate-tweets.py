
import pandas as pd
import numpy as np
import glob
from transformers import pipeline, MarianTokenizer, MarianMTModel

MODEL_PATH = 'Helsinki-NLP/opus-mt-'

models = {
    'it': MODEL_PATH+'it-en',
    'tr': MODEL_PATH+'tr-en',
    'de': MODEL_PATH+'de-en',
    'sv': MODEL_PATH+'sv-en',
    'da': MODEL_PATH+'da-en',
    'es': MODEL_PATH+'es-en',
    'fr': MODEL_PATH+'fr-en',
    'nl': MODEL_PATH+'nl-en',
    'hu': MODEL_PATH+'hu-en',
    'fi': MODEL_PATH+'fi-en',
    'ar': MODEL_PATH+'ar-en',
    'ur': MODEL_PATH+'ur-en',
    'ja': MODEL_PATH+'ja-en',
    'ru': MODEL_PATH+'ru-en',
    'pl': MODEL_PATH+'pl-en',
    'et': MODEL_PATH+'et-en',
    'ca': MODEL_PATH+'ca-en',
    'cy': MODEL_PATH+'cy-en',
    'is': MODEL_PATH+'is-en',
    'eu': MODEL_PATH+'eu-en',
    'ht': MODEL_PATH+'ht-en',
    'zh': MODEL_PATH+'zh-en',
}

# Languages for which no translation model found:
# in, lt, ps, pt, ro, sl, no, ca, eu

HEALTH_ORGS_PATH = '../../data/health-organizations/'
LEADERS_PATH = '../../data/leaders/'

# Iterate through all the files in a folder and translate the non-english tweets
for file in glob.glob(HEALTH_ORGS_PATH+'/*.csv'):
    user_df = pd.read_csv(file)
    username = user_df['username'].unique()[0]
    print(username, user_df.shape)    
    
    for index, row in user_df.iterrows():
        language = row['language']
        input_tw = row['tweet']   
        # Translate tweet if language is not english and if we have a model for that
        if language != 'en' and language in models.keys():    
            model = MarianMTModel.from_pretrained(models[language])
            tokenizer = MarianTokenizer.from_pretrained(models[language])
            translation_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
            tweet_en = translation_pipeline(input_tw)[0]['generated_text']
            user_df.at[index, 'tweet_en'] = tweet_en
    
    user_df.to_csv(LEADERS_PATH+'/'+username+'.csv', index=False)

df = user_df[user_df['tweet_en'].notnull()]

df.to_csv(HEALTH_ORGS_PATH+username+'.csv', index=False)




