# OIBSIP-DataAnalystics-L2-Task3-Fraud-Detection-
# Credit Card Fraud Detection Using Machine Learning

## Project Overview

This project aims to detect fraudulent credit card transactions using machine learning classification algorithms. Financial fraud detection is challenging because fraudulent transactions represent only a small percentage of all transactions. Therefore, the project focuses on handling class imbalance and evaluating models using appropriate classification metrics rather than relying on accuracy alone.

The project uses the Credit Card Fraud Detection dataset and compares Logistic Regression and Random Forest classifiers. SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training data to help address the imbalance between legitimate and fraudulent transactions.

## Objectives

- Analyze the distribution of legitimate and fraudulent transactions.
- Perform exploratory data analysis (EDA) on transaction amounts and time features.
- Understand why accuracy can be misleading for imbalanced datasets.
- Apply SMOTE oversampling to address class imbalance.
- Train Logistic Regression and Random Forest classification models.
- Evaluate models using precision, recall, F1-score, ROC-AUC, and average precision.
- Visualize confusion matrices and ROC and Precision-Recall curves.
- Analyze feature importance and model coefficients.
- Discuss scalability for high-volume financial transaction processing.

## Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Imbalanced Data Handling:** Imbalanced-learn (SMOTE)
- **Development Environment:** Jupyter Notebook

## Dataset

The project uses the Credit Card Fraud Detection dataset available on Kaggle.

**Dataset link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The benchmark dataset contains 284,807 transactions, including 492 fraudulent transactions. It has 30 input features and one target column named `Class`.

- `Class = 0`: Legitimate transaction
- `Class = 1`: Fraudulent transaction

The features include transaction time, amount, and anonymized PCA components (`V1` to `V28`).

## Project Workflow

1. Load the credit card transaction dataset.
2. Inspect data types, missing values, duplicates, and class distribution.
3. Perform exploratory data analysis on transaction amounts and time.
4. Visualize the class imbalance and transaction patterns.
5. Split the dataset into training and testing sets using stratified sampling.
6. Apply SMOTE only to the training data.
7. Train Logistic Regression and Random Forest classifiers.
8. Evaluate model performance using classification metrics.
9. Plot confusion matrices, ROC curves, and Precision-Recall curves.
10. Analyze feature importance and Logistic Regression coefficients.
11. Compare the models and discuss scalability considerations.

## Machine Learning Models

### 1. Logistic Regression

Logistic Regression estimates the probability that a transaction is fraudulent. It provides a useful baseline model and allows coefficient analysis to understand how input features affect predictions.

### 2. Random Forest Classifier

Random Forest combines multiple decision trees to identify complex patterns in transaction data. It can model nonlinear relationships and provides feature importance scores.

## Handling Class Imbalance

Fraudulent transactions represent approximately 0.173% of the original benchmark dataset. This imbalance can cause a model to favor the majority class.

SMOTE generates synthetic minority-class examples to help the model learn patterns associated with fraud. In this project, SMOTE is applied only to the training data to avoid contaminating the test set.

## Model Evaluation

The models are evaluated using:

- **Precision:** Measures how many transactions flagged as fraudulent are actually fraudulent.
- **Recall:** Measures how many actual fraudulent transactions are detected.
- **F1-score:** Balances precision and recall.
- **ROC-AUC:** Measures the model's ability to rank fraudulent transactions above legitimate ones across different thresholds.
- **Average Precision:** Summarizes the Precision-Recall curve and is useful for highly imbalanced classification.
- **Confusion Matrix:** Shows true positives, false positives, true negatives, and false negatives.

Accuracy is included for reference but is not the primary success metric because a model can achieve high accuracy while missing most fraudulent transactions.

## Installation

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

## How to Run

1. Download the dataset from Kaggle.
2. Extract `creditcard.csv` into the project directory.
3. Open Jupyter Notebook.
4. Open `fraud_detection.ipynb`.
5. Execute the notebook cells in order.
6. Review the evaluation metrics, charts, feature analysis, and model comparison.

## Expected Outputs

- Fraud versus legitimate transaction distribution chart.
- Transaction amount distribution plots.
- Approximate elapsed-hour fraud analysis.
- Model performance comparison table.
- Classification reports and confusion matrices.
- ROC and Precision-Recall curves.
- Random Forest feature importance chart.
- Logistic Regression coefficient analysis.
- CSV files containing model metrics, predictions, and feature importance.

## Scalability Considerations

A production fraud detection system may need to process millions of transactions per hour. Scalability can be improved through efficient preprocessing, optimized model inference, horizontally scalable prediction services, stream-processing systems, and continuous model monitoring. Time-based validation and retraining are also important because fraud patterns change over time.

## Conclusion

This project demonstrates how machine learning can be used to identify potentially fraudulent credit card transactions in a highly imbalanced dataset. By combining SMOTE with Logistic Regression and Random Forest, the project explores methods for improving minority-class detection. Evaluation using precision, recall, F1-score, ROC-AUC, and average precision provides a more meaningful assessment than accuracy alone. The results can support further work on threshold optimization, fraud monitoring, and scalable transaction risk assessment.

## References

- Kaggle Credit Card Fraud Detection: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Imbalanced-learn SMOTE documentation: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
- Scikit-learn documentation: https://scikit-learn.org/stable/
- Pandas documentation: https://pandas.pydata.org/docs/
- Seaborn documentation: https://seaborn.pydata.org/
