# Credit-Card-Fraud-Detection-Model
Exploratory analysis and predictive modeling on [Credit Card Detection Model] from Kaggle, including feature engineering and model evaluation.


Credit Card Fraud Detection — Machine Learning on an Imbalanced Dataset

This project builds and evaluates multiple machine‑learning models to detect fraudulent credit card transactions in a highly imbalanced real‑world dataset. The goal is to develop a reliable fraud‑detection approach that maximizes fraud capture while minimizing false positives — a critical balance for financial institutions.

Dataset Overview
The dataset contains 284,807 transactions made by European cardholders in September 2013, of which only 492 are fraudulent (≈0.17%).
Key characteristics:
- Features V1–V28 are PCA‑transformed components
- Time and Amount are the only original features
- Class is the target variable (1 = fraud, 0 = non‑fraud)
- No missing values
- Extreme class imbalance: 99.83% non‑fraud vs. 0.17% fraud
A processed version of this dataset is publicly available on Kaggle.
The original research was conducted by Worldline and the Machine Learning Group of ULB.

Exploratory Data Analysis
- No missing values were found.
- Fraud cases represent only 0.17% of all transactions.
- Time and Amount are heavily skewed and require scaling.
- PCA‑transformed features show meaningful correlations with fraud:
- Positively correlated: V2, V4, V11, V19
- Negatively correlated: V10, V12, V14, V17
These correlations help guide model behavior despite anonymized features.

Data Preprocessing
- Scaled Amount and Time using standard scaling.
- Removed outliers from features strongly correlated with fraud using the IQR method.
- Ensured that all resampling (SMOTE, NearMiss) was applied only to the training fold inside cross‑validation to avoid data leakage.

Modeling and Evaluation

Given the extreme class imbalance, both undersampling and oversampling approaches were tested.

1. Undersampling with NearMiss
Models tested: Logistic Regression, Random Forest, XGBoost
Best performer: Logistic Regression
Metrics:
- Accuracy: 0.7683
- Precision: 0.0066
- Recall: 0.9560
- F1: 0.0131
- AUC: 0.9477

Interpretation:
High recall but extremely low precision — too many false positives for real‑world use.

2. Oversampling with SMOTE
Metrics:
- Accuracy: 0.9721
- Precision: 0.0479
- Recall: 0.8681
- F1: 0.0908
- ROC AUC: 0.9607
- PR AUC (AP): 0.6686
  
Interpretation:
Better balance than undersampling, but precision remains low at the default threshold.

3. Neural Network + SMOTE (Best Model)
Metrics:
- Accuracy: 0.9996
- Precision: 0.9351
- Recall: 0.7912
- F1: 0.8571
- ROC AUC: 0.9648
- PR AUC (AP): 0.8657
  
Interpretation:
This model achieves excellent precision and strong recall, with the highest PR AUC among all tested models.
It provides the best balance for operational fraud detection.

Key Takeaways
- Accuracy is not meaningful for imbalanced datasets.
- PR AUC, Precision, Recall, and F1 are the appropriate metrics.
- SMOTE improves recall but may introduce noise.
- NearMiss increases recall but drastically reduces precision.
- Neural Network + SMOTE delivers the strongest overall performance.
- Threshold tuning is essential to balance fraud capture vs. false alarms.
- Real‑world fraud detection requires minimizing both financial loss and customer friction.

Technologies Used
- Python
- Pandas, NumPy
- Scikit‑learn
- Imbalanced‑learn (SMOTE, NearMiss)
- XGBoost
- TensorFlow / Keras
- Matplotlib, Seaborn
