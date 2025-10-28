# creditcard-fraud-detection-model
A machine learning model using KNN for classifying transactions as fraud or genuine after feature engineering and scaling.

🕵️‍♂️ Fraud Detection Using K-Nearest Neighbors (KNN)

This project focuses on building a Fraud Detection System using the K-Nearest Neighbors (KNN) algorithm. The goal is to classify transactions as fraudulent or genuine based on real-world transaction data.
It demonstrates the complete machine learning workflow — from data preprocessing and feature engineering to model training and evaluation.

🚀 Project Overview

Fraud detection is a critical task for financial institutions to prevent losses and protect customers.
In this project, a K-Nearest Neighbors (KNN) model is implemented using scikit-learn to classify fraudulent transactions.
The dataset includes various attributes like transaction amount, merchant name, and transaction time.

The project pipeline includes:

Data Loading & Exploration

Feature Engineering (extracting hour and day from timestamp)

Categorical Encoding using LabelEncoder

Data Normalization using StandardScaler

Model Training using KNN

Performance Evaluation using Accuracy Score

🧠 Technologies Used

Python 

Pandas – Data manipulation

NumPy – Numerical operations

Scikit-learn (sklearn) – Machine Learning algorithms and preprocessing tools

Google Colab / Jupyter Notebook – Development environment


🔍 Future Improvements

Handle class imbalance using SMOTE or undersampling techniques

Perform hyperparameter tuning (GridSearchCV) for optimal k value

Try alternative models like:

Random Forest

Logistic Regression

XGBoost

Add precision, recall, and confusion matrix analysis for better evaluation



