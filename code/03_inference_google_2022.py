import pandas as pd
import numpy as np
from joblib import load

# Inputs
path_inference_data = "../party_classifiers/g2022_adid_text.csv.gz"
path_inference_data_vars = "../party_classifiers/g2022_adid_var1.csv.gz"

df = pd.read_csv(path_inference_data)
var = pd.read_csv(path_inference_data_vars)

df = df.merge(var[['ad_id', 'advertiser_id']], on='ad_id', how='left')


# Replace newline characters with spaces
df['aws_ocr_video_text'] = df['aws_ocr_video_text'].str.replace('\\n', ' ')
df['ad_text'] = df['ad_text'].str.replace('\\n', ' ')
df['aws_ocr_img_text'] = df['aws_ocr_img_text'].str.replace('\\n', ' ')
df['ad_title'] = df['ad_title'].str.replace('\\n', ' ')
df['google_asr_text'] = df['google_asr_text'].str.replace('\\n', ' ')

cols = ['advertiser_name', 'ad_title', 'ad_text', 'google_asr_text', 'aws_ocr_img_text', 
        'aws_ocr_video_text']
df['combined'] = df[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
df['combined'] = df['combined'].str.strip()

df_txt = df.groupby(['advertiser_id'])['combined'].apply(lambda x: ' '.join(x)).reset_index()

# Load the best model weights
mnb_clf = load('../models/party_clf_pdid_mnb.joblib')

pred = mnb_clf.predict(df_txt['combined'])

df_txt['party_all_clf'] = pred
df_txt = df_txt[['advertiser_id', 'party_all_clf']]

df_txt.to_csv("../data/party_all_clf_google_2022_advertiser_id.csv.gz", index=False, compression = 'gzip')

