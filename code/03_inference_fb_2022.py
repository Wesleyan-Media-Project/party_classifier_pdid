import pandas as pd
from joblib import load

# Output of "data-post-production repo", available on Wesleyan Media Project Figshare
d = pd.read_csv('fb_2022_adid_text.csv.gz')
pdid = pd.read_csv('fb_2022_adid_var1.csv.gz', usecols=['ad_id', 'pd_id'])
d = d.merge(pdid, on='ad_id', how='left')

cols = ['page_name', 'disclaimer', 'ad_creative_body', 'google_asr_text', 'aws_ocr_text_img', 'aws_ocr_text_vid', 
        'ad_creative_link_caption', 'ad_creative_link_title', 'ad_creative_link_description']
d['combined'] = d[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
d['combined'] = d['combined'].str.strip()

# Deduplicate before concatenating ad texts
dd = d.drop_duplicates(subset=['pd_id','combined'], keep='last')

d_pdid_txt = dd.groupby(['pd_id'])['combined'].apply(lambda x: ' '.join(x)).reset_index()

# Load the best model weights
mnb_clf = load('../models/party_clf_pdid_mnb.joblib')

pred = mnb_clf.predict(d_pdid_txt['combined'])

d_pdid_txt['party_all_clf_pdid'] = pred
d_pdid_txt = d_pdid_txt[['pd_id', 'party_all_clf_pdid']]

d_pdid_txt.to_csv("../data/party_all_clf_pdid_fb_2022.csv.gz", index=False, compression='gzip')
