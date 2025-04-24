#Liver Disease Prediction Using Machine Learning

This repository contains the complete implementation of a machine learning project designed to predict liver disease using the Indian Liver Patient Dataset. The project uses both classification (SVM) and regression (Random Forest) models to analyze key liver biomarkers and aid in clinical decision-making.

📊 Project Overview
Liver disease is a critical global health concern, and early detection is essential. This project applies machine learning to assist in:

Classifying patients as having liver disease or not using a Support Vector Machine (SVM).

Predicting bilirubin levels using Random Forest Regression to estimate disease severity.

📁 Dataset
Source: Kaggle - Indian Liver Patient Dataset

Size: 583 records

Features:

Demographics: Age, Gender

Biochemical indicators: Bilirubin (Total/Direct), Liver Enzymes (ALT, AST, ALP)

Protein markers: Albumin, Albumin/Globulin ratio

Target: Liver disease presence (binary)

🧪 EDA Highlights
Males are 2.7× more likely to have liver disease than females

Peak disease incidence observed in the 45–60 age group

Strong correlations between liver enzyme levels and disease status

🧠 Machine Learning Models
1. Support Vector Machine (SVM)
Purpose: Classification

Accuracy: 72%

Recall (Disease class): 88%

Top Features: AST, Albumin, ALP

2. Random Forest Regression
Purpose: Predicting Total Bilirubin levels

R² Score: 0.68

MAE: 1.92 mg/dL

🔍 Feature Importance (Top 5)

Feature	SVM Weight	RF Importance
AST (SGOT)	0.22	0.23
Albumin	0.19	0.21
ALP	0.15	0.17
Age	0.11	0.09
Total Proteins	0.08	0.07
⚙️ Technical Stack
Python

Pandas, NumPy

Matplotlib, Seaborn

scikit-learn

📈 Future Scope
Model Improvement: Deep learning, hybrid models with imaging

Deployment: EHR-integrated decision support systems

Fairness & Ethics: Bias audits, federated learning for privacy

📄 Report
The full project report is included in the repository as a PDF, detailing methodology, results, and future recommendations.

🧑‍🎓 Author
Kumud Ranjan
M.Tech Data Science and Engineering
Lovely Professional University
