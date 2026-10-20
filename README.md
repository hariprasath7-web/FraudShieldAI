# FraudShield AI — Intelligent Fraud Detection System

## Overview
FraudShield AI is a machine-learning project that analyses credit-card transaction data and identifies transactions that may require fraud review.

## Technologies Used
- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Random Forest Classifier
- Isolation Forest Anomaly Detection

## Key Features
- Dark-themed dashboard with red and green risk indicators
- Dataset upload and validation
- Model training and evaluation
- Precision, recall, F1-score, PR-AUC and confusion matrix
- Transaction risk scoring
- Anomaly detection
- CSV prediction report download

## Live Demo
https://fraudshieldai-irfg6kwoocafob8olzmdky.streamlit.app/

## Run Locally
1. Install Python.
2. Install dependencies using `pip install -r requirements.txt`.
3. Place the labelled dataset at `data/creditcard.csv`.
4. Run `python -m streamlit run app.py`.

## Limitations
This is an educational prototype. Model predictions are not proof of fraud, and the system should not be used alone to make real payment decisions.
