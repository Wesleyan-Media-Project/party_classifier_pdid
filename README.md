# party_classifier_pdid

This party clf is trained at the entity level using a MultinomialNB model.

### Steps
1. Keep entities in the 1.18m dataset which have 'party_all' info from wmp_fb_entities_v051822.csv
2. Split entities into train and test (train_size = 0.8)
3. Prepare text for training, test, and inference:
    3.1 Deduplicate using 'pd_id' and 'ad_combined_text'
    3.2 Concatenate ads' 'ad_combined_text' for each pd id
4. Training models at the entity level - MultinomialNB, Logistic regression, and SVM
5. Pick the best model based on classification reports.
6. Make inference: party_clf_pdid_mnb.joblib has been applied to the 1.18m dataset (entity_level/concatenated ad ad_combined_text)

### Model details
GridSearchCV best parameters:
- 'clf__alpha': 0.01
- 'tfidf__norm': 'l2',
- 'tfidf__use_idf': False,
- 'vect__ngram_range': (2, 2)

### Model performance
Performance on held-out test set:
```
              precision    recall  f1-score   support

DEM           0.862         0.956     0.907       274
OTHER         1.000         0.091     0.167        22
REP           0.900         0.855     0.877       200

accuracy                             0.877       496
macro avg     0.921        0.634     0.650       496
weighted avg  0.883        0.877     0.862       496
```

### Output files (Files can be downloaded from Google Drive)
* Training results
    - `/content/drive/Shareddrives/Delta Lab/Data/facebook_2020_party_all_pdid/party_clf_pdid_mnb.joblib`
    - `/content/drive/Shareddrives/Delta Lab/Data/facebook_2020_party_all_pdid/party_clf_pdid_logit.joblib`
    - `/content/drive/Shareddrives/Delta Lab/Data/facebook_2020_party_all_pdid/party_clf_pdid_svm.joblib`

* results
    - `/content/drive/Shareddrives/Delta Lab/Data/facebook_2020_party_all_pdid/party_clf_entity_fb_118m.csv`
