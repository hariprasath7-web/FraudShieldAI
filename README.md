# FraudShield AI

An educational AI-based transaction fraud-risk detection
prototype using Random Forest and Isolation Forest.

## Features

- Supervised fraud classification
- Unsupervised anomaly detection
- Interactive Streamlit dashboard
- Precision, recall, F1-score and PR-AUC
- Confusion matrix
- CSV risk report export

## Requirements

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Local Setup

1. Install Python.
2. Open the project folder in VS Code.
3. Create and activate a virtual environment.
4. Install dependencies:

   python -m pip install -r requirements.txt

5. Place the permitted public dataset at:

   data/creditcard.csv

6. Start the app:

   python -m streamlit run app.py

## Limitations

This project is an educational prototype.
A risk score is not proof of fraud. Do not use it alone
to block payments or make financial decisions.

Public deployment requires privacy protection, security
testing, independent model validation and human review.
