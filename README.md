
# Machine Learning Assignment - 2
## Breast Cancer Classification using Machine Learning

---

# 1. Problem Statement

The objective of this assignment is to develop and compare multiple
machine learning classification models using a common classification
dataset.

The models are trained on the selected dataset and evaluated using six
performance metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

An interactive Streamlit application is also developed to allow users
to upload test data, select a machine learning model, and view its
evaluation results.

---

# 2. Dataset Description

## Dataset

**Breast Cancer Wisconsin (Diagnostic) Dataset**

The dataset is a binary classification dataset used to predict whether
a breast tumor is:

- Benign
- Malignant

## Dataset Characteristics

| Property | Details |
|---|---|
| Problem Type | Binary Classification |
| Total Instances | 569 |
| Number of Features | 30 |
| Target Variable | Diagnosis |
| Classes | Benign (B), Malignant (M) |
| Training Samples | 455 |
| Testing Samples | 114 |
| Test Split | 20% |
| Random State | 42 |

The dataset satisfies the assignment requirement of at least 500
instances and 12 features.

## Preprocessing

The target variable was encoded as:

- B = 0 (Benign)
- M = 1 (Malignant)

The dataset was divided into training and testing sets using an 80:20
split with stratification.

Feature scaling was applied to the models that require scaled
features:

- Logistic Regression
- k-Nearest Neighbors (kNN)

The scaler was fitted only on the training data and then applied to
the test data.

---

# 3. GitHub Repository Link

**GitHub Repository:**

(https://github.com/kunwarvishal18/ML-Assignment-2-Breast-Cancer)

The repository contains:

```text
project-folder/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── scaler.pkl
```


---

# 4. Models Used

The following machine learning classification models were implemented
and evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (kNN)
4. Naive Bayes
5. Random Forest (Ensemble)

All models were evaluated using the same test dataset and the following
six performance metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 4.1 Model Comparison

The performance of all five machine learning models on the test dataset
is shown below.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| kNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest (Ensemble) | 0.9649 | 0.9942 | 1.0000 | 0.9048 | 0.9500 | 0.9258 |

---

## 4.2 Observations on Model Performance

### Logistic Regression

Logistic Regression achieved an accuracy of **0.9649** and an AUC of
**0.9960**.

It achieved the highest recall among the evaluated models at
**0.9286**, indicating strong performance in identifying malignant
cases.

Its other performance scores were:

- Precision: **0.9750**
- F1 Score: **0.9512**
- MCC: **0.9245**

Overall, Logistic Regression demonstrated strong and balanced
classification performance.

---

### Decision Tree

Decision Tree achieved an accuracy of **0.9298** and an AUC of
**0.9246**.

Its performance scores were:

- Precision: **0.9048**
- Recall: **0.9048**
- F1 Score: **0.9048**
- MCC: **0.8492**

Among the five evaluated models, Decision Tree produced the lowest
overall performance across the main evaluation metrics.

---

### K-Nearest Neighbors (kNN)

kNN achieved an accuracy of **0.9561** and an AUC of **0.9823**.

Its performance scores were:

- Precision: **0.9744**
- Recall: **0.9048**
- F1 Score: **0.9383**
- MCC: **0.9058**

The model demonstrated strong overall classification performance,
although its recall was lower than Logistic Regression.

---

### Naive Bayes

Naive Bayes achieved an accuracy of **0.9386** and an AUC of
**0.9934**.

The model achieved perfect precision:

**Precision = 1.0000**

Its other performance scores were:

- Recall: **0.8333**
- F1 Score: **0.9091**
- MCC: **0.8715**

Although Naive Bayes achieved a very high AUC and perfect precision,
its recall was the lowest among the evaluated models.

---

### Random Forest (Ensemble)

Random Forest achieved an accuracy of **0.9649**, matching Logistic
Regression for the highest accuracy.

Its performance scores were:

- AUC: **0.9942**
- Precision: **1.0000**
- Recall: **0.9048**
- F1 Score: **0.9500**
- MCC: **0.9258**

Random Forest achieved perfect precision and the highest MCC among the
evaluated models.

Its recall was slightly lower than Logistic Regression.

Overall, Random Forest demonstrated excellent classification
performance and was very close to Logistic Regression.

---

## 4.3 Overall Winner

### Overall Winner: Logistic Regression

Based on the evaluation results, **Logistic Regression** was selected
as the overall winner.

Its performance was:

| Metric | Score |
|---|---:|
| Accuracy | 0.9649 |
| AUC | 0.9960 |
| Precision | 0.9750 |
| Recall | 0.9286 |
| F1 Score | 0.9512 |
| MCC | 0.9245 |

Logistic Regression achieved:

- Joint-highest Accuracy of **0.9649**
- Highest AUC of **0.9960**
- Highest Recall of **0.9286**
- F1 Score of **0.9512**
- MCC of **0.9245**

Random Forest was very close, with the same accuracy of 0.9649,
perfect precision of 1.0000, and the highest MCC of 0.9258.

However, Logistic Regression achieved the highest AUC and recall and
provided the strongest overall balance across the six evaluation
metrics.

Therefore, **Logistic Regression was selected as the Overall Winner.**

---

# 5. Streamlit Application

An interactive Streamlit application was developed to demonstrate
the trained machine learning classification models.

The application provides the following functionality:

- Upload test data in CSV format
- Select a machine learning model
- Display Accuracy
- Display AUC
- Display Precision
- Display Recall
- Display F1 Score
- Display MCC
- Display Confusion Matrix
- Display Classification Report
- Display Prediction Results

## Models Available in the Application

The Streamlit application supports:

1. Logistic Regression
2. Decision Tree
3. kNN
4. Naive Bayes
5. Random Forest

## Streamlit Application Link

**Live Streamlit App:**

(https://ml-assignment-2-breast-cancer-sjniyoxvuquvrabh3x3bqx.streamlit.app/#4-evaluation-results-logistic-regression)

---

# 6. Test Data

The file `test_data.csv` contains the test data used to evaluate all
the trained machine learning models.

The test dataset contains:

- **114 test instances**
- **30 input features**
- **1 target column: Diagnosis**

Therefore, the test CSV contains:

```text
114 rows
31 columns
```

---

# 7. Requirements

All the project's dependencies are listed in `requirements.txt`.

```text
streamlit
pandas
numpy
scikit-learn
joblib
```

---

# 8. Project Structure

The project repository has the following structure:

```text
project-folder/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── scaler.pkl
```

---

# 9. Conclusion

This project successfully implemented and evaluated five machine
learning classification models for breast cancer diagnosis. Logistic
Regression emerged as the top-performing model, demonstrating a strong
balance of accuracy, AUC, and recall.

The Streamlit application provides an interactive platform for model
evaluation, allowing users to upload new data and gain insights into
the models' performance.
