# Liver Disease Prediction Using Machine Learning

This project leverages machine learning to predict liver disease using clinical data from the **Indian Liver Patient Dataset**. It combines exploratory data analysis (EDA), classification, and regression modeling to extract meaningful healthcare insights.

## 📂 Dataset

- **Source**: [Kaggle - Indian Liver Patient Dataset](https://www.kaggle.com/datasets/uciml/indian-liver-patient-records)
- **Records**: 583
- **Features**: Age, Gender, Bilirubin levels, Liver enzymes (ALT, AST, ALP), Albumin, A/G ratio
- **Target Variable**: Liver Disease (0 = No Disease, 1 = Disease)

## 🔍 Project Objectives

- Perform EDA to understand patterns in liver disease indicators
- Compare healthy vs. diseased patient data statistically and visually
- Build:
  - **SVM** model for disease classification
  - **Random Forest** model for bilirubin level prediction

## 🧪 Exploratory Data Analysis (EDA)

- Analyzed gender and age-wise distribution
- Investigated enzyme and protein level variations
- Identified strong correlations using heatmaps
- Handled missing values and encoded categorical data

## 🧠 Machine Learning Models

### 1. Support Vector Machine (SVM)
- **Task**: Binary Classification
- **Accuracy**: 72%
- **Recall (Disease)**: 88%
- **Precision**: 75%
- **F1-Score**: 0.81
- **Top Features**: AST, Albumin, ALP

### 2. Random Forest Regressor
- **Task**: Predicting Total Bilirubin
- **R² Score**: 0.68
- **MAE**: 1.92 mg/dL
- **RMSE**: 2.89 mg/dL

### 📊 Feature Importance (Top 5)
| Feature         | SVM Weight | RF Importance |
|-----------------|------------|---------------|
| AST (SGOT)      | 0.22       | 0.23          |
| Albumin         | 0.19       | 0.21          |
| ALP             | 0.15       | 0.17          |
| Age             | 0.11       | 0.09          |
| Total Proteins  | 0.08       | 0.07          |

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn

## 🧠 Key Insights

- Males are 2.7× more likely to develop liver disease
- Peak disease incidence observed between ages 45–60
- High bilirubin and low albumin levels strongly indicate disease

## 🚀 Future Scope

- Integrate deep learning (CNNs) for image+biochemical analysis
- Deploy models as EHR-integrated decision support tools
- Incorporate federated learning for secure, cross-hospital collaboration
- Apply fairness audits to detect and mitigate bias

## 📄 Report

The full project report is included as `Liver_Disease_Prediction_Using_Machine_Learning_Report.pdf`.

## 👤 Author

**Kumud Ranjan**  
M.Tech Data Science and Engineering  
Lovely Professional University
