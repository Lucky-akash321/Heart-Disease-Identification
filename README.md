# Heart Disease Identification Using Machine Learning

![](https://github.com/Lucky-akash321/Heart-Disease-Identification/blob/main/HD.png)

## Introduction
Heart disease is one of the leading causes of mortality worldwide. **Early identification and prediction of heart disease** using **machine learning models** can help in timely medical intervention. This guide provides a step-by-step approach to building a **heart disease prediction model** using machine learning.

### Dataset Source
We will use the **Heart Disease UCI Dataset** from **Kaggle**:
🔗 **Dataset Link**: [Heart Disease UCI - Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci)  

---

## Step 1: Understanding the Dataset
### 1.1 Features in the Dataset
The dataset contains **14 attributes** used to predict the presence of heart disease. The key attributes include:

- **Age**: Age of the patient.
- **Sex**: Gender (1 = Male, 0 = Female).
- **Chest Pain Type (cp)**: 
  - 0: Typical angina  
  - 1: Atypical angina  
  - 2: Non-anginal pain  
  - 3: Asymptomatic  
- **Resting Blood Pressure (trestbps)**: Blood pressure in mm Hg.
- **Cholesterol (chol)**: Serum cholesterol level.
- **Fasting Blood Sugar (fbs)**: (1 = True, 0 = False).
- **Resting ECG (restecg)**: ECG results (0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy).
- **Max Heart Rate (thalach)**: Maximum heart rate achieved.
- **Exercise-Induced Angina (exang)**: (1 = Yes, 0 = No).
- **ST Depression (oldpeak)**: ST depression induced by exercise.
- **Slope of ST Segment (slope)**: (0 = Upsloping, 1 = Flat, 2 = Downsloping).
- **Number of Major Vessels (ca)**: Count of major vessels (0-3).
- **Thalassemia (thal)**: (0 = Normal, 1 = Fixed defect, 2 = Reversible defect).
- **Target Variable**: (1 = Presence of heart disease, 0 = No heart disease).

---

## Step 2: Data Preprocessing
### 2.1 Data Cleaning
- Remove duplicate values (if any).
- Handle missing values using **mean/median imputation** for numerical features.

### 2.2 Encoding Categorical Variables
- Convert **categorical features** (e.g., `cp`, `thal`, `slope`) into numerical values using **One-Hot Encoding** or **Label Encoding**.

### 2.3 Feature Scaling
- Standardize numerical features using **Min-Max Scaling** or **Standardization (Z-score normalization)**.

---

## Step 3: Exploratory Data Analysis (EDA)
### 3.1 Visualizing Data Distribution
- Use **histograms** to analyze age distribution.
- Plot **correlation heatmaps** to identify feature relationships.
- Use **boxplots** to detect outliers in `chol`, `trestbps`, and `thalach`.

### 3.2 Checking Class Imbalance
- **Target variable distribution**: If imbalanced, apply **SMOTE (Synthetic Minority Over-sampling Technique)**.

---

## Step 4: Splitting Data for Training & Testing
- **80% Training Set, 20% Testing Set** for model evaluation.
- Use **Stratified Sampling** to maintain class balance.

---

## Step 5: Building Machine Learning Models
### 5.1 Selecting Algorithms
Different ML models can be used for heart disease prediction:
1. **Logistic Regression** (Baseline Model)
2. **Random Forest Classifier**
3. **Support Vector Machine (SVM)**
4. **Gradient Boosting (XGBoost, LightGBM)**
5. **Artificial Neural Networks (ANNs)**

### 5.2 Training the Model
- Fit selected models on **training data**.
- Optimize using **hyperparameter tuning (GridSearchCV, RandomizedSearchCV)**.

---

## Step 6: Model Evaluation
### 6.1 Performance Metrics
Use the following metrics for evaluating model accuracy:
- **Accuracy**: `(TP + TN) / (TP + TN + FP + FN)`
- **Precision, Recall, and F1-score** to measure class performance.
- **ROC-AUC Curve** for model discrimination ability.
- **Confusion Matrix** for analyzing false positives & false negatives.

### 6.2 Comparing Models
- Evaluate multiple models and select the best **based on precision-recall balance**.
- If **Random Forest/XGBoost performs better**, use **Feature Importance Analysis** to interpret key predictors.

---

## Step 7: Deploying the Model
### 7.1 Saving the Trained Model
- Save the final model using **Pickle (.pkl) or Joblib**.

### 7.2 Deploying as a Web App
- Use **Flask/FastAPI** to create an API endpoint for real-time predictions.
- Deploy on **AWS, Google Cloud, or Heroku**.

---

## Step 8: Continuous Improvement
- **Monitor Model Performance**: Use **new data updates** for retraining.
- **Explainability & Interpretability**: Use **SHAP (SHapley Additive Explanations)** to understand feature importance.
- **Enhance Performance**: Try **deep learning models** (ANN, CNN for ECG analysis).

---

## Conclusion
Building a **Heart Disease Prediction System** using **Machine Learning** enables **early detection and better decision-making in healthcare**. By following these steps, we can create a robust, scalable model for real-world medical applications.

🔗 **Dataset**: [Heart Disease UCI - Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci)  
🚀 **Next Steps**: Enhance the model using **Deep Learning (CNN/LSTM for ECG data)**.

---
