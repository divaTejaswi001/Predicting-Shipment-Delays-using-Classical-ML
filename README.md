#  Shipment Delay Prediction using Classical Machine Learning

This project predicts whether a shipment will be **delayed or on-time** using classical machine learning techniques on structured logistics data.

It demonstrates a complete ML pipeline including:
- Data preprocessing and feature engineering
- Handling class imbalance with SMOTE
- Model training and hyperparameter tuning using GridSearchCV
- Model evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
- Feature selection and interpretability

---

## Key Features

- **Data Source**: Customer Shipment dataset (logistics domain)
- **Tech Stack**: Python, Scikit-learn, XGBoost, Pandas, Seaborn, SMOTE, GridSearchCV
- **ML Models Used**:
  - Random Forest (with class weighting)
- **Feature Engineering**:
  - Log transformation of skewed features
  - Bucketization of weight
  - One-hot encoding of categorical variables
- **Evaluation Metrics**:
  - Confusion Matrix
  - Classification Report
  - ROC-AUC Score
- **Cross-validation**:
  - 5-fold cross-validation on tuned models

---

##  Results

- **ROC-AUC Score**: ~0.75
- **Best Model**: Random Forest with SMOTE and tuned hyperparameters
- **Observations**:
  - High precision for delay prediction (~96%)
  - Good separation between on-time and delayed classes with adjusted threshold

---



