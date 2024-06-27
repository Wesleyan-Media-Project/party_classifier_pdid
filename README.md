# Wesleyan Media Project - Party Classifier with Unique ID

Welcome! This repository contains scripts that train and apply a machine learning model to classify political advertisements based on their content and determine which political party (Democratic, Republican, or Other) the ads belong to.

This repo is part of the [Cross-platform Election Advertising Transparency Initiative (CREATIVE)](https://www.creativewmp.com/). CREATIVE is an academic research project that has the goal of providing the public with analysis tools for more transparency of political ads across online platforms. In particular, CREATIVE provides cross-platform integration and standardization of political ads collected from Google and Facebook. CREATIVE is a joint project of the [Wesleyan Media Project (WMP)](https://mediaproject.wesleyan.edu/) and the [privacy-tech-lab](https://privacytechlab.org/) at [Wesleyan University](https://www.wesleyan.edu).

To analyze the different dimensions of political ad transparency we have developed an analysis pipeline. The scripts in this repo are part of the Data Classification step in our pipeline.

![A picture of the repo pipeline with this repo highlighted](Creative_Pipelines.png)

## Table of Contents

- [1. Overview](#1-overview)
  - [Model details](#model-details)
  - [Model performance](#model-performance)
- [2. Setup](#2-setup)
- [3. Thank You](#3-thank-you)

## 1. Overview

We provide 4 different party classifier models that are each trained using different algorithms. We recommend (and implement in our scripts) the Random Forest model which provided the highest accuracy in our training. Nonetheless, we give you all four models should you wish to use them. The party classifier is trained on a dataset of ads that have already been labeled with the party each ad belongs to.

In addition to the classifier in this repo, we also provide an [ad-level party classifier](https://github.com/Wesleyan-Media-Project/party_classifier). Unlike the ad-level party classifier that operates at the individual ad level, the party classifier in this repo works at the entity level by analyzing all ads associated with a particular entity ["pd_id"](https://github.com/Wesleyan-Media-Project/fb_pd_id) collectively.

In situations where you need clear and specific predictions about political party affiliations for ads, it is better to use this party classifier instead of the ad-level classifier because the former operates under the assumption that all ads associated with a single pd_id will belong to the same party, leading to more consistent and potentially more accurate predictions about party affiliation when viewing ads collectively rather than individually.

## 2 Setup

The scripts in this repo are in [Python](https://www.python.org/). Make sure you have both installed and set up before continuing. To install and set up Python you can follow the [Beginner's Guide to Python](https://wiki.python.org/moin/BeginnersGuide). The Python Scripts in this repo uses Jupyter Notebook as an interface. It is an interactive environment for Python development. You can install Jupyter Notebook by following the [Jupyter Notebook website](https://jupyter.org/install).

To start setting up the repo and run the scripts, first clone this repo to your local directory:

```bash
git clone https://github.com/Wesleyan-Media-Project/party_classifier_pdid.git
```

Then, ensure you have the following software packages installed for Python:

- `scikit-learn`
- `pandas`
- `numpy`
- `joblib`

You can install the required packages by running the following command:

```bash
pip install pandas numpy scikit-learn joblib
```

After installing the required packages, you can run the scripts in the following order:

1. `01_create_training_data.ipynb`
2. `02_training.ipynb`
3. `03_google2022_inference.ipynb`, `03_inference_140m.ipynb`, or `03_inference_fb2022.ipynb`

To run the above IPython Notebook code ending with `.ipynb`, you can open the Jupyter Notebook interface by type the following in your terminal:

```bash
jupyter notebook
```

After you open the Jupyter Notebook interface, you can navigate to the folder where you have cloned the repo and open the script you want to run.

Then, click on the first code cell to select it.
Run each cell sequentially by clicking the Run button or pressing `Shift + Enter`.

If you want to use the trained model we provide, you can only run the inference script since the model files are already present in the [`/models`](https://github.com/Wesleyan-Media-Project/party_classifier_pdid/blob/main/models) folder.

### 2.1 Training

Note: If you do not want to train a model from scratch, you can use the trained model we provide [here](https://github.com/Wesleyan-Media-Project/party_classifier_pdid/blob/main/models/party_clf_pdid_rf.joblib), and skip to 2.2.

To run our scripts, you need to have a trained classifier. The script [`01_create_training_data.ipynb`](https://github.com/Wesleyan-Media-Project/party_classifier_pdid/blob/main/01_create_training_data.ipynb) prepares a training dataset by first reading the ad data `fb_2020_140m_adid_var1.csv.gz` (ADD FIGSHARE LINK ONCE READY) which has the metadata for each ad and merges it with the WMP entity file [`wmp_fb_entities_v090622.csv`](https://github.com/Wesleyan-Media-Project/datasets/blob/main/wmp_entity_files/Facebook/2022/wmp_fb_2022_entities_v091922.csv) which has the party affiliation information on each entity that publishes ads on Facebook, based on the pd_id column. This allows the script to associate each ad with a party affiliation.

Second, the script checks for each page ID (page_id) and ensures all associated ads have consistent party affiliations. If a page ID has ads with conflicting party affiliations, it marks that page ID as non-usable.

Third, the script split page names into train and test with a 70/30 split to make sure pd ids of a usable page into either train or test but never both.

Finally, the script prepares text for train, testing, and inference by filtering out the rows with non-usable page IDs and saving the resulting data frame as a compressed CSV file: [`140m_with_page_id_based_training_data.csv.gz`](https://github.com/Wesleyan-Media-Project/party_classifier_pdid/blob/main/data/facebook/140m_with_page_id_based_training_data.csv.gz). This file contains the ad data, page IDs, party affiliations, and the train-test split information for the usable page IDs.

Once our training data is ready, the script [`02_training.ipynb`](https://github.com/Wesleyan-Media-Project/party_classifier_pdid/blob/main/02_training.ipynb) loads this training data and trains the party classifier. During the training process, the script trains the following machine learning models and picks the best model (e.g., Random Forest) based on classification reports:

- Multinomial Naive Bayes (MultinomialNB)
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

All models are saved in the [`/models`](https://github.com/Wesleyan-Media-Project/party_classifier_pdid/blob/main/models) folder. The best model which later used to inference the new data is [`party_clf_pdid_rf.joblib`](https://github.com/Wesleyan-Media-Project/party_classifier_pdid/blob/main/models/party_clf_pdid_rf.joblib).

#### 2.1.1 Model details

The best-performing model is identified as the Random Forest Classifier with the following hyperparameters we found using Grid Search Cross-Validation:

- 'clfmax_depth': 25,
- 'clfmax_features': 0.1,
- 'clf\_\_n_estimators': 500

#### 2.1.2 Model performance

Here is the model performance on the held-out test set:

```
              precision    recall  f1-score   support

         DEM      0.843     0.941     0.889       491
       OTHER      1.000     0.091     0.167        44
         REP      0.887     0.851     0.869       424

    accuracy                          0.862       959
   macro avg      0.910     0.628     0.642       959
weighted avg      0.870     0.862     0.847       959
```

### 2.2 Inference

After the training, the following scripts `03_google2022_inference.ipynb`, `03_inference_140m.ipynb`, and `03_inference_fb2022.ipynb` are all used to apply the trained model to different datasets. The applied output is saved accordingly in the file `party_all_clf_google_2022_advertiser_id.csv`, `party_all_clf_pdid_fb_2020_140m.csv`, and `party_all_clf_pdid_fb_2022.csv` respectively. Here are the input files you need for each of these inference scripts:

- For Facebook 2020: `fb_2020_140m_adid_text_clean.csv.gz` and `fb_2020_140m_adid_var1.csv.gz` (ADD FIGSHARE LINK ONCE READY)
- For Google 2022: `g2022_adid_01062021_11082022_text.csv.gz` (ADD FIGSHARE LINK ONCE READY)
- For Facebook 2022: `fb_2022_adid_text.csv.gz` and `fb_2022_adid_var1.csv.gz` (ADD FIGSHARE LINK ONCE READY)

Note: If you would like to use a model different than Random Forest, you can simply change the model input script with the appropriate model. For instance, if you want to use the SVM model, replace the following script in the inference scripts:
`mnb_clf = load('models/party_clf_pdid_rf.joblib')`
with this:
`mnb_clf = load('models/party_clf_pdid_svm.joblib')`

## 3. Thank You

<p align="center"><strong>We would like to thank our supporters!</strong></p><br>

<p align="center">This material is based upon work supported by the National Science Foundation under Grant Numbers 2235006, 2235007, and 2235008.</p>

<p align="center" style="display: flex; justify-content: center; align-items: center;">
  <a href="https://www.nsf.gov/awardsearch/showAward?AWD_ID=2235006">
    <img class="img-fluid" src="nsf.png" height="150px" alt="National Science Foundation Logo">
  </a>
</p>

<p align="center">The Cross-Platform Election Advertising Transparency Initiative (CREATIVE) is a joint infrastructure project of the Wesleyan Media Project and privacy-tech-lab at Wesleyan University in Connecticut.

<p align="center" style="display: flex; justify-content: center; align-items: center;">
  <a href="https://www.creativewmp.com/">
    <img class="img-fluid" src="CREATIVE_logo.png"  width="220px" alt="CREATIVE Logo">
  </a>
</p>

<p align="center" style="display: flex; justify-content: center; align-items: center;">
  <a href="https://mediaproject.wesleyan.edu/">
    <img src="wmp-logo.png" width="218px" height="100px" alt="Wesleyan Media Project logo">
  </a>
</p>

<p align="center" style="display: flex; justify-content: center; align-items: center;">
  <a href="https://privacytechlab.org/" style="margin-right: 20px;">
    <img src="./plt_logo.png" width="200px" alt="privacy-tech-lab logo">
  </a>
</p>
