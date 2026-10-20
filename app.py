
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# DARK + RED + GREEN THEME
# =========================================================
st.markdown("""
<style>
.stApp {
    background-color: #090B10;
    color: #F8FAFC;
}
[data-testid="stHeader"] {
    background-color: #090B10;
}
section[data-testid="stSidebar"] {
    background-color: #11141B;
    border-right: 1px solid #303541;
}
h1, h2, h3 {
    color: #F8FAFC !important;
}
p, label, li {
    color: #D1D5DB;
}
[data-testid="stMetric"] {
    background: #151922;
    border: 1px solid #303541;
    border-radius: 14px;
    padding: 16px;
}
.stButton > button {
    background: linear-gradient(100deg, #B91C1C, #15803D);
    color: white;
    border: 1px solid #374151;
    border-radius: 10px;
    font-weight: 700;
    min-height: 44px;
}
.stButton > button:hover {
    border: 1px solid #4ADE80;
    color: white;
}
div[data-testid="stFileUploader"] {
    background: #11141B;
    border: 1px dashed #4B5563;
    border-radius: 12px;
    padding: 12px;
}
[data-testid="stDataFrame"] {
    border: 1px solid #303541;
    border-radius: 10px;
}
hr {
    border-color: #303541;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div style="
    background: linear-gradient(120deg, #151922, #251116);
    border: 1px solid #7F1D1D;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;">
    <h1 style="margin:0;color:#FFFFFF;">
        🛡️ FraudShield AI
    </h1>
    <p style="margin-bottom:0;color:#D1D5DB;">
        AI-Powered Credit Card Fraud Detection Dashboard
    </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Educational prototype | Random Forest + Isolation Forest"
)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🛡️ FraudShield AI")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "Choose a page",
        ["Dashboard", "Train & Evaluate", "Analyse Transactions", "About"],
    )
    st.markdown("---")
    st.markdown("### Risk legend")
    st.markdown("🔴 **High risk:** needs manual review")
    st.markdown("🟢 **Low risk:** no high-risk signal detected")
    st.caption("AI predictions are not proof of fraud.")

# =========================================================
# DATA LOADING AND VALIDATION
# =========================================================
DATA_PATH = Path("data") / "creditcard.csv"


def load_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        return None


def validate_training_data(data):
    if data is None or data.empty:
        return False, "The dataset is empty."

    if "Class" not in data.columns:
        return False, "Training CSV must contain a 'Class' column (0=legitimate, 1=fraud)."

    if data.columns.duplicated().any():
        return False, "The CSV contains duplicate column names."

    if data["Class"].isna().any():
        return False, "The Class column contains missing labels."

    labels = set(data["Class"].unique())
    if not labels.issubset({0, 1, 0.0, 1.0}):
        return False, "Class must contain only 0 and 1."

    if data["Class"].nunique() < 2:
        return False, "Training needs both legitimate and fraud examples."

    features = data.drop(columns=["Class"])
    if features.empty:
        return False, "No feature columns were found."

    if not all(pd.api.types.is_numeric_dtype(features[c]) for c in features.columns):
        return False, "All transaction feature columns must be numeric."

    if np.isinf(features.to_numpy(dtype=float)).any():
        return False, "Feature data contains infinity values."

    return True, "Dataset looks valid."


def train_models(data):
    feature_names = [c for c in data.columns if c != "Class"]
    X = data[feature_names].copy()
    y = data["Class"].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Supervised fraud classifier
    rf_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    rf_model.fit(X_train, y_train)
    probabilities = rf_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    # Anomaly detector trained on legitimate training transactions only
    normal_train = X_train[y_train == 0]
    if len(normal_train) < 10:
        raise ValueError("Not enough legitimate training examples for anomaly detection.")

    isolation_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("detector", IsolationForest(
            n_estimators=100,
            contamination="auto",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    isolation_model.fit(normal_train)
    normal_scores = -isolation_model.decision_function(normal_train)
    anomaly_threshold = float(np.quantile(normal_scores, 0.99))

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "pr_auc": average_precision_score(y_test, probabilities),
        "confusion": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Fraud"],
            zero_division=0,
            output_dict=True,
        ),
        "test_labels": y_test,
        "test_predictions": predictions,
    }

    return {
        "rf_model": rf_model,
        "isolation_model": isolation_model,
        "features": feature_names,
        "threshold": anomaly_threshold,
        "metrics": metrics,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }


# =========================================================
# CHOOSE TRAINING DATA
# =========================================================
if page in ["Dashboard", "Train & Evaluate"]:
    st.markdown("## 📂 Training Dataset")

    uploaded_training = st.file_uploader(
        "Optional: upload a labelled training CSV",
        type=["csv"],
        key="training_csv",
        help="Must contain numeric transaction features and a Class column.",
    )

    data = None
    if uploaded_training is not None:
        data = load_csv(uploaded_training)
    elif DATA_PATH.exists():
        try:
            data = pd.read_csv(DATA_PATH)
            st.info("Loaded local dataset: data/creditcard.csv")
        except Exception as exc:
            st.error(f"Could not load local dataset: {exc}")
    else:
        st.warning(
            "Dataset not found. Create a data folder, put creditcard.csv "
            "inside it, or upload a labelled CSV above."
        )

    if data is not None:
        valid, message = validate_training_data(data)
        if valid:
            fraud_count = int((data["Class"] == 1).sum())
            normal_count = int((data["Class"] == 0).sum())

            c1, c2, c3 = st.columns(3)
            c1.metric("Total transactions", f"{len(data):,}")
            c2.metric("Legitimate transactions", f"{normal_count:,}")
            c3.metric("Fraud examples", f"{fraud_count:,}")

            st.success(message)
            with st.expander("Preview training data"):
                st.dataframe(data.head(10), use_container_width=True)
        else:
            st.error(message)
            data = None

    if st.button("🚀 Train and Evaluate AI Models", use_container_width=True):
        if data is None:
            st.error("Provide a valid labelled dataset first.")
        else:
            try:
                with st.spinner("Training models. Large datasets may take a while..."):
                    st.session_state["fraudshield"] = train_models(data)
                st.success("Models trained successfully!")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

# =========================================================
# MODEL EVALUATION
# =========================================================
if page == "Train & Evaluate":
    st.markdown("## 📊 Model Evaluation")

    saved = st.session_state.get("fraudshield")
    if not saved:
        st.info("First load a dataset and click Train and Evaluate AI Models.")
    else:
        m = saved["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m['accuracy']:.2%}")
        c2.metric("Precision", f"{m['precision']:.2%}")
        c3.metric("Recall", f"{m['recall']:.2%}")

        c4, c5 = st.columns(2)
        c4.metric("F1 Score", f"{m['f1']:.2%}")
        c5.metric("Average Precision (PR-AUC)", f"{m['pr_auc']:.4f}")

        st.markdown("### Confusion Matrix")
        cm = pd.DataFrame(
            m["confusion"],
            index=["Actual Legitimate", "Actual Fraud"],
            columns=["Predicted Legitimate", "Predicted Fraud"],
        )
        st.dataframe(cm, use_container_width=True)

        st.markdown("### Classification Report")
        report_df = pd.DataFrame(m["report"]).transpose()
        st.dataframe(report_df, use_container_width=True)

        st.caption(
            f"Training rows: {saved['train_rows']:,} | "
            f"Test rows: {saved['test_rows']:,}"
        )

# =========================================================
# TRANSACTION ANALYSIS
# =========================================================
if page == "Analyse Transactions":
    st.markdown("## 🔎 Analyse New Transactions")
    st.write(
        "Upload a CSV containing the same feature columns used during training. "
        "Do not include the Class label."
    )

    saved = st.session_state.get("fraudshield")

    if not saved:
        st.warning(
            "No trained model in this session. Go to Dashboard, load a dataset, "
            "and click Train and Evaluate AI Models first."
        )
    else:
        uploaded_transactions = st.file_uploader(
            "Upload transactions CSV",
            type=["csv"],
            key="transactions_csv",
        )

        if uploaded_transactions is not None:
            try:
                new_data = pd.read_csv(uploaded_transactions)

                if new_data.empty:
                    st.error("The uploaded CSV is empty.")
                elif "Class" in new_data.columns:
                    st.error(
                        "Remove the Class column from this prediction CSV. "
                        "Class labels are for training/evaluation."
                    )
                elif new_data.columns.duplicated().any():
                    st.error("The CSV contains duplicate column names.")
                else:
                    expected = saved["features"]
                    missing = [c for c in expected if c not in new_data.columns]
                    extra = [c for c in new_data.columns if c not in expected]

                    if missing or extra:
                        if missing:
                            st.error(f"Missing required columns: {missing[:10]}")
                        if extra:
                            st.error(f"Unexpected columns: {extra[:10]}")
                        st.info(
                            "The prediction CSV must use the same feature columns "
                            "as the training dataset."
                        )
                    elif not all(
                        pd.api.types.is_numeric_dtype(new_data[c])
                        for c in expected
                    ):
                        st.error("All transaction feature columns must be numeric.")
                    elif np.isinf(new_data[expected].to_numpy(dtype=float)).any():
                        st.error("The CSV contains infinity values.")
                    else:
                        X_new = new_data[expected].copy()
                        model = saved["rf_model"]
                        isolation = saved["isolation_model"]

                        risk_scores = model.predict_proba(X_new)[:, 1]
                        risk_flags = np.where(
                            risk_scores >= 0.50,
                            "High Risk",
                            "Low Risk",
                        )

                        anomaly_scores = -isolation.decision_function(X_new)
                        anomaly_flags = np.where(
                            anomaly_scores > saved["threshold"],
                            "Anomaly",
                            "No anomaly signal",
                        )

                        results = new_data.copy()
                        results["fraud_risk_score"] = risk_scores
                        results["risk_flag"] = risk_flags
                        results["anomaly_score"] = anomaly_scores
                        results["anomaly_flag"] = anomaly_flags

                        high_risk_count = int((risk_flags == "High Risk").sum())
                        low_risk_count = int((risk_flags == "Low Risk").sum())

                        a, b, c = st.columns(3)
                        a.metric("Transactions analysed", f"{len(results):,}")
                        b.metric("🔴 High risk", f"{high_risk_count:,}")
                        c.metric("🟢 Low risk", f"{low_risk_count:,}")

                        st.markdown("### Prediction Results")
                        st.dataframe(results, use_container_width=True)

                        if high_risk_count:
                            st.error(
                                f"🚨 {high_risk_count} transaction(s) flagged "
                                "for manual review."
                            )
                        else:
                            st.success(
                                "No transactions crossed the current high-risk threshold."
                            )

                        csv_bytes = results.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download Prediction Report",
                            data=csv_bytes,
                            file_name="fraudshield_predictions.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

            except Exception as exc:
                st.error(f"Could not analyse this CSV: {exc}")

# =========================================================
# DASHBOARD SUMMARY
# =========================================================
if page == "Dashboard":
    st.markdown("## 🧠 AI System Overview")

    saved = st.session_state.get("fraudshield")

    if saved:
        st.success("Models are trained and ready for analysis in this session.")
        m = saved["metrics"]

        x1, x2, x3 = st.columns(3)
        x1.metric("Model F1 Score", f"{m['f1']:.2%}")
        x2.metric("Model Recall", f"{m['recall']:.2%}")
        x3.metric("Average Precision", f"{m['pr_auc']:.4f}")
    else:
        st.info("Load the dataset above and train the models to start.")

    st.markdown("### Models used")
    st.markdown("""
    - **Random Forest:** supervised classifier that estimates fraud risk.
    - **Isolation Forest:** anomaly detector that identifies unusual transactions.
    - **Risk score:** model-estimated fraud probability, not a guarantee.
    """)

# =========================================================
# ABOUT
# =========================================================
if page == "About":
    st.markdown("## ℹ️ About FraudShield AI")

    st.write("""
    FraudShield AI is an educational machine-learning prototype for credit-card
    transaction analysis. It combines a supervised Random Forest classifier
    with an unsupervised Isolation Forest anomaly detector.
    """)

    st.markdown("### Important limitations")
    st.warning("""
    - A high-risk flag is not proof that a transaction is fraudulent.
    - A low-risk score does not guarantee a transaction is safe.
    - The dataset may contain anonymized features and class imbalance.
    - Evaluate false positives, false negatives, and data drift before any real use.
    - Do not use this prototype alone to block payments or accuse customers.
    """)

st.markdown("---")
st.caption("FraudShield AI | Educational prototype | Dark • Red • Green")
