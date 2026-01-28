# YouTube Comment Spam Classifier (NLP)

This project implements a machine learning–based system to classify YouTube comments
as **spam** or **non-spam (ham)** using classical Natural Language Processing (NLP) techniques.

---

## 🔗 Project Links

- **Live Demo (Hugging Face Space):**  
  https://huggingface.co/spaces/leyuzak/YouTube-Comment-Spam-Classifier-NLP

- **Kaggle Notebook:**  
  https://www.kaggle.com/code/leyuzakoksoken/youtube-spam-collection-classification

---

## 🎯 Project Objective

The primary objective of this project is to build, evaluate, and deploy
a text classification model capable of detecting spam comments on YouTube.

The project focuses on:
- Applying classical machine learning models for text classification
- Comparing multiple models using standard evaluation metrics
- Deploying the best-performing model as a web application

---

## 📊 Dataset Description

The dataset used in this project is the **YouTube Spam Collection Dataset**.

- Total samples: 1,956 YouTube comments  
- Labels:
  - `0` → Non-spam (ham)
  - `1` → Spam  
- Source: Multiple popular YouTube videos  

The dataset is nearly balanced, which supports effective supervised learning.

---

## 🔍 Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the dataset structure
and class distribution prior to model training.

The following analyses were conducted:
- Dataset shape and structure inspection
- Missing value analysis
- Class distribution visualization

These steps ensured that the dataset was clean and suitable for classification tasks.

---

## 🧹 Data Preprocessing

The following preprocessing steps were applied:

- Selection of relevant columns
- Renaming columns for consistency
- Handling missing values
- Splitting data into features and labels
- Transforming text data using **TF-IDF Vectorization**

Minimal text preprocessing was applied, as TF-IDF effectively captures
important textual patterns in sparse text data.

---

## 🤖 Model Training

Three machine learning models were trained and evaluated:

- Logistic Regression
- Multinomial Naive Bayes
- Linear Support Vector Machine (Linear SVM)

The dataset was split using an **80/20 train-test split**.
TF-IDF features were used as input for all models.

---

## 📊 Results Comparison

Models were evaluated using the following metrics:

- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)

### Model Performance Summary

| Model              | Accuracy |
|--------------------|----------|
| Linear SVM         | ~0.95    |
| Logistic Regression | ~0.93    |
| Multinomial Naive Bayes | ~0.89 |

Linear SVM achieved the best overall performance across all evaluation metrics.

---

## 🔎 Confusion Matrix Analysis

Confusion matrices were generated for all trained models to analyze
classification behavior and misclassifications.

Linear SVM produced the lowest number of classification errors,
confirming its effectiveness for spam detection.

---

## 🚀 Deployment

The best-performing model (**Linear SVM**) was deployed as a web application
using **Streamlit** and hosted on **Hugging Face Spaces**.

- The trained model and TF-IDF vectorizer were saved and reused during deployment
- The application allows users to input a YouTube comment and receive a prediction in real time

---

## 🏁 Conclusion

This project demonstrates that classical machine learning models,
when combined with TF-IDF vectorization, can effectively detect spam
in user-generated text.

Among all evaluated models, Linear SVM showed the strongest performance
and was selected for deployment.

---

## 🔮 Future Work

- Experiment with deep learning models (LSTM, BERT)
- Add language detection and multilingual support
- Integrate the model into content moderation pipelines
