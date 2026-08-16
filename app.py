
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("📊 Breast Cancer Classification Dashboard")

st.markdown(
    """
    This application evaluates multiple machine learning
    classification models using the Breast Cancer Wisconsin
    Diagnostic dataset.
    """
)

st.info(
    "Upload the test CSV file and select a machine learning model "
    "to view its classification performance."
)


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {
        "Logistic Regression":
            joblib.load("model/logistic_regression.pkl"),

        "Decision Tree":
            joblib.load("model/decision_tree.pkl"),

        "kNN":
            joblib.load("model/knn.pkl"),

        "Naive Bayes":
            joblib.load("model/naive_bayes.pkl"),

        "Random Forest":
            joblib.load("model/random_forest.pkl")
    }

    scaler = joblib.load("model/scaler.pkl")

    return models, scaler


# ============================================================
# LOAD MODELS SAFELY
# ============================================================

try:

    models, scaler = load_models()

except Exception as e:

    st.error(
        "Unable to load the trained model files."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# MODEL SELECTION
# ============================================================

st.subheader("1. Select Machine Learning Model")

selected_model_name = st.selectbox(
    "Choose a model:",
    list(models.keys())
)

selected_model = models[selected_model_name]


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("2. Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


# ============================================================
# MAIN PROCESSING
# ============================================================

if uploaded_file is None:

    st.warning(
        "Please upload the test_data.csv file to evaluate the model."
    )

    st.stop()


# ============================================================
# READ CSV
# ============================================================

try:

    data = pd.read_csv(uploaded_file)

except Exception as e:

    st.error("Unable to read the uploaded CSV file.")

    st.code(str(e))

    st.stop()


# ============================================================
# DISPLAY UPLOADED DATA
# ============================================================

st.subheader("3. Uploaded Test Data")

st.write(
    f"Dataset contains **{data.shape[0]} rows** "
    f"and **{data.shape[1]} columns**."
)

st.dataframe(
    data.head(10),
    use_container_width=True
)


# ============================================================
# CHECK TARGET COLUMN
# ============================================================

target_column = "Diagnosis"

if target_column not in data.columns:

    st.error(
        "The uploaded CSV must contain a 'Diagnosis' target column."
    )

    st.write("Columns found in uploaded file:")

    st.write(
        list(data.columns)
    )

    st.stop()


# ============================================================
# IDENTIFY EXPECTED FEATURE COLUMNS
# ============================================================

try:

    expected_features = list(
        scaler.feature_names_in_
    )

except AttributeError:

    expected_features = [
        column
        for column in data.columns
        if column != target_column
    ]


# ============================================================
# CHECK FEATURE COLUMNS
# ============================================================

uploaded_features = [
    column
    for column in data.columns
    if column != target_column
]


missing_features = [
    feature
    for feature in expected_features
    if feature not in uploaded_features
]


extra_features = [
    feature
    for feature in uploaded_features
    if feature not in expected_features
]


if missing_features:

    st.error(
        "The uploaded CSV is missing required feature columns."
    )

    st.write(
        "Missing columns:"
    )

    st.write(missing_features)

    st.stop()


if extra_features:

    st.warning(
        "The uploaded CSV contains additional columns. "
        "Only the trained model features will be used."
    )


# ============================================================
# SELECT FEATURES IN EXACT TRAINING ORDER
# ============================================================

X = data[expected_features].copy()

y = data[target_column].copy()


# ============================================================
# CONVERT FEATURES TO NUMERIC
# ============================================================

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


# ============================================================
# CHECK MISSING FEATURE VALUES
# ============================================================

missing_value_count = X.isnull().sum().sum()

if missing_value_count > 0:

    st.warning(
        f"{missing_value_count} missing feature values were found. "
        "Median imputation will be applied."
    )

    X = X.fillna(
        X.median()
    )


# ============================================================
# CONVERT TARGET
# ============================================================

if y.dtype == "object":

    y = y.map(
        {
            "B": 0,
            "M": 1,
            "Benign": 0,
            "Malignant": 1
        }
    )


# ============================================================
# CHECK TARGET VALUES
# ============================================================

if y.isnull().any():

    st.error(
        "The Diagnosis column contains unsupported target values."
    )

    st.write(
        "Expected values: B/M or Benign/Malignant"
    )

    st.stop()


y = y.astype(int)


# ============================================================
# PREPARE INPUT DATA
# ============================================================

if selected_model_name in [
    "Logistic Regression",
    "kNN"
]:

    X_input = scaler.transform(X)

else:

    X_input = X


# ============================================================
# MAKE PREDICTIONS
# ============================================================

try:

    y_pred = selected_model.predict(
        X_input
    )

    y_prob = selected_model.predict_proba(
        X_input
    )[:, 1]

except Exception as e:

    st.error(
        "An error occurred while generating predictions."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y,
    y_pred
)

precision = precision_score(
    y,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y,
    y_pred,
    zero_division=0
)

mcc = matthews_corrcoef(
    y,
    y_pred
)


# ============================================================
# AUC
# ============================================================

try:

    auc = roc_auc_score(
        y,
        y_prob
    )

    auc_display = f"{auc:.4f}"

except ValueError:

    auc = None

    auc_display = "N/A"


# ============================================================
# DISPLAY MODEL NAME
# ============================================================

st.divider()

st.subheader(
    f"4. Evaluation Results — {selected_model_name}"
)


# ============================================================
# DISPLAY METRICS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

with col2:

    st.metric(
        "AUC",
        auc_display
    )

with col3:

    st.metric(
        "Precision",
        f"{precision:.4f}"
    )


col4, col5, col6 = st.columns(3)

with col4:

    st.metric(
        "Recall",
        f"{recall:.4f}"
    )

with col5:

    st.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

with col6:

    st.metric(
        "MCC",
        f"{mcc:.4f}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader("5. Confusion Matrix")

cm = confusion_matrix(
    y,
    y_pred
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual Benign",
        "Actual Malignant"
    ],
    columns=[
        "Predicted Benign",
        "Predicted Malignant"
    ]
)

st.dataframe(
    cm_df,
    use_container_width=True
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.subheader("6. Classification Report")

report = classification_report(
    y,
    y_pred,
    target_names=[
        "Benign",
        "Malignant"
    ],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# ============================================================
# PREDICTION RESULTS
# ============================================================

st.subheader("7. Prediction Results")

prediction_data = data.copy()

prediction_data[
    "Predicted Diagnosis"
] = np.where(
    y_pred == 1,
    "Malignant",
    "Benign"
)

st.dataframe(
    prediction_data,
    use_container_width=True
)
