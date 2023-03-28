# party_classifier_pdid

This party clf is trained at the entity level using a Random Forest model.

### Steps
1. Keep entities in the 1.40m dataset which have 'party_all' info from wmp_fb_entities_v051822.csv
2. Split page names into train and test (train_size = 0.7) to make sure pd ids of a page into either train or test but never both. 
3. Prepare text for train, test, and inference
4. Training models at the entity level - MultinomialNB, Logistic regression, SVM, and Random Forest
5. Pick the best model based on classification reports
6. Make inference: party_clf_pdid_rf.joblib has been applied to the 1.40m dataset (entity_level/concatenated ad combined text)

### Model details
GridSearchCV best Params:  {'clf__max_depth': 25, 'clf__max_features': 0.1, 'clf__n_estimators': 500}

### Model performance
Performance on held-out test set:
```
              precision    recall  f1-score   support

         DEM      0.843     0.941     0.889       491
       OTHER      1.000     0.091     0.167        44
         REP      0.887     0.851     0.869       424

    accuracy                          0.862       959
   macro avg      0.910     0.628     0.642       959
weighted avg      0.870     0.862     0.847       959
```

### Output files (Large files that need to be downloaded from Google Drive)
* Model weights
    - `/content/drive/Shareddrives/Delta Lab/github/party_classifier_pdid /party_clf_pdid_mnb.joblib`

