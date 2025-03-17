import sklearn.model_selection as ms
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

if __name__ == '__main__':
    # Input
    path_train_test = "../data/party_pdid_train_prepared.csv.gz"

    # Read in the data
    df = pd.read_csv(path_train_test, encoding='UTF-8')

    # Split
    train, test = ms.train_test_split(df, test_size=0.2, stratify=df.party_all, random_state=1802)

    # Models
    np.random.seed(1802)


    # Logistic regression
    print('Logistic regression')

    lr_clf = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', LogisticRegression(max_iter=500)),
                   ])


    lr_params = [{'clf__penalty': ['l2'],
            'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'clf__solver': ['newton-cg']}] 

    lr_grid = GridSearchCV(lr_clf, lr_params, cv=5, n_jobs=-10) 
    lr_grid.fit(train['text'], train['party_all'])

    print("Best Params: ", lr_grid.best_params_)
    print(metrics.classification_report(test['party_all'], lr_grid.predict(test['text']), digits = 3))

    dump(lr_grid, '../models/party_clf_pdid_logit.joblib')


    ## SVM
    print('SVM')

    svm_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier())])

    svm_params = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-3),
    }

    svm_grid = GridSearchCV(svm_clf, svm_params, cv=5, n_jobs=-10)
    svm_grid.fit(train['text'], train['party_all'])

    print("Best Params: ", svm_grid.best_params_)
    print(metrics.classification_report(test['party_all'], svm_grid.predict(test['text']), digits = 3))

    dump(svm_grid, '../models/party_clf_pdid_svm.joblib')

    # MultinomialNB
    print('Multinomial NB')

    mnb_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    mnb_params = {
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': [1, 1e-1, 1e-2]
    }

    mnb_grid = GridSearchCV(mnb_clf, mnb_params, cv=5, n_jobs=-10)
    mnb_grid.fit(train['text'], train['party_all'])

    print("Best Params: ", mnb_grid.best_params_)
    print(metrics.classification_report(test['party_all'], mnb_grid.predict(test['text']), digits = 3))

    dump(mnb_grid, '../models/party_clf_pdid_mnb.joblib')

    # Random Forest
    print('Random Forest')
    rf_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ("clf", RandomForestClassifier())
    ])

    rf_params = {
        'clf__n_estimators': [100,200,300,400,500],
        'clf__max_depth': [15, 20, 25, 30, 35, 40],
    }

    rf_grid = GridSearchCV(rf_clf, rf_params, cv=5, n_jobs=4)
    rf_grid.fit(train['text'], train['party_all'])

    print("Best Params: ", rf_grid.best_params_)
    print(metrics.classification_report(test['party_all'], rf_grid.predict(test['text']), digits = 3))

    dump(rf_grid, '../models/party_clf_pdid_rf.joblib')




