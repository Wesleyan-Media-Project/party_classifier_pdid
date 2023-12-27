# Wesleyan Media Project - Party Classifier with Unique ID

Welcome! This repo is a part of the Cross-platform Election Advertising Transparency initiative ([CREATIVE](https://www.creativewmp.com/)) project. CREATIVE is a joint infrastructure project of WMP and privacy-tech-lab at Wesleyan University. CREATIVE provides cross-platform integration and standardization of political ads collected from Google and Facebook. You will also need the repo [datasets](https://github.com/Wesleyan-Media-Project/datasets), [fb_2020](https://github.com/Wesleyan-Media-Project/fb_2020) and [data-post-production](https://github.com/Wesleyan-Media-Project/data-post-production) repos to run the script.

This repo is a part of the Final Data Classification step.
![A picture of the repo pipeline with this repo highlighted](Creative_Pipelines.png)

## Table of Contents

- [Introduction](#introduction)
- [Objective](#objective)
- [Data](#data)
- [Setup](#setup)
  - [Requirements](#requirements)
  - [Repo Workflow](#repo-workflow)
  - [Model details](#model-details)
  - [Model performance](#model-performance)

## Introduction

This party classifier is trained at the entity level using a Random Forest model. It concatenates all ads of a pd_id into one. In practice, when discrete party predictions are needed, you should choose this classifier compared to [ad-level one](https://github.com/Wesleyan-Media-Project/party_classifier), because it assumes that all ads belonging to a pd_id will be the same party.

## Objective

Each of our repos belongs to one or more of the following categories:

- Data Collection
- Data Storage & Processing
- Preliminary Data Classification
- Final Data Classification

This repo is part of the Final Data Classification section.

## Data

The data created by the scripts in this repo is in csv.gz format and located in `/data` folder. The training model is saved in `/models` folder. There is one training data `party_clf_pdid_mnb.joblib` that is too large to be uploaded to Github. You may download it by your own from our Figshare.

## Setup

The scripts are numbered in the order in which they should be run. Scripts that directly depend on one another are ordered sequentially. Scripts with the same number are alternatives; usually they are the same scripts for different data, or with minor variations. The outputs of each script are saved, so it is possible to, for example, only run the inference script, since the model files are already present.

### Requirements

Here is the breakdown of the packages you will need to install to run the scripts in this repo:

- `scikit-learn`
- `pandas`
- `numpy`
- `joblib`

You can type the following command in your terminal:

```bash
pip install pandas numpy scikit-learn joblib
```

to install the packages.

### Repo Workflow

1. Keep entities in the 1.40m dataset which have 'party_all' info from wmp_fb_entities_v051822.csv
2. Split page names into train and test (train_size = 0.7) to make sure pd ids of a page into either train or test but never both.
3. Prepare text for train, test, and inference
4. Training models at the entity level - MultinomialNB, Logistic regression, SVM, and Random Forest
5. Pick the best model based on classification reports
6. Make inference: party_clf_pdid_rf.joblib has been applied to the 1.40m dataset (entity_level/concatenated ad combined text)

### Model details

GridSearchCV best Params: {'clf**max_depth': 25, 'clf**max_features': 0.1, 'clf\_\_n_estimators': 500}

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
