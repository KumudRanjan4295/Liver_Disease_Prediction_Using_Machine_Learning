# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 12:59:26 2025

@author: bgpda
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

# Load the data
df = pd.read_csv("C:/Users/bgpda/Desktop/LPU/Python_DataScience/indian_liver_patient.csv")

# Print Data
print(df)
print(df.shape)
# Basic Information
print(df.info())
print(df.head())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Handle missing values (only one in Albumin_and_Globulin_Ratio)
df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].median(), inplace=True)

# Convert Gender to numerical (Male=1, Female=0)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Rename target forr clarity
df = df.rename(columns={'Dataset':'Liver_Disease'})

# Check target distribution
print(df['Liver_Disease'].value_counts())

########################################### ANALYSIS ###################################################

# 1. Gender Distribution Analysis: Examine the proportion of males and females in the dataset and their liver disease status.

gender_analysis = df.groupby(['Gender', 'Liver_Disease']).size().unstack()
gender_analysis.columns = ['No Disease', 'Disease']
gender_analysis.index = ['Female', 'Male']

plt.figure(figsize=(8,5))
gender_analysis.plot(kind='bar', stacked=True)
plt.title('Liver Disease Cases by Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Findings:
# (a) Males are significantly overrepresented in the dataset
# (b) Males have a much higher incidence of liver disease compared to females


# 2. Age Distribution by Disease Status Analysis: Compare age distributions between patients with and without liver disease.

plt.figure(figsize=(10,6))
sns.violinplot(x='Liver_Disease', y='Age', data=df, split=True)
plt.title('Age Distribution by Liver Disease Status')
plt.xticks([0,1], ['Disease', 'No Disease'])
plt.show()

# Findings:
# (a) Patients with liver disease tend to be slightly older
# (b) The age range is similar for both groups (20-70 years)
# (c) Younger patients (<20) are more likely to not have liver disease


# 3. Bilirubin Levels Analysis: Compare bilirubin levels between healthy and diseased patients.

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.boxplot(x='Liver_Disease', y='Total_Bilirubin', data=df)
plt.title('Total Bilirubin by Disease Status')

plt.subplot(1,2,2)
sns.boxplot(x='Liver_Disease', y='Direct_Bilirubin', data=df)
plt.title('Direct Bilirubin by Disease Status')
plt.tight_layout()
plt.show()

# Findings:
# (a) Both total and direct bilirubin levels are significantly higher in patients with liver disease
# (b) Many outliers in the diseased group indicate severe cases


# 4. Liver Enzymes Analysis: Compare key liver enzymes (ALT, AST, ALP) between groups.

enzymes = ['Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase']

plt.figure(figsize=(15,5))
for i, enzyme in enumerate(enzymes, 1):
    plt.subplot(1,3,i)
    sns.boxplot(x='Liver_Disease', y=enzyme, data=df)
    plt.title(f'{enzyme} by Disease Status')
plt.tight_layout()
plt.show()

# Findings:
# (a) All three enzymes show significantly higher levels in diseased patients
# (b) AST shows the most dramatic difference between groups
# (c) Extreme outliers suggest some acute liver injury cases


# 5. Protein Levels Analysis: Examine protein-related biomarkers (Total Proteins, Albumin, A/G Ratio).

proteins = ['Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']

plt.figure(figsize=(15,5))
for i, protein in enumerate(proteins, 1):
    plt.subplot(1,3,i)
    sns.boxplot(x='Liver_Disease', y=protein, data=df)
    plt.title(f'{protein} by Disease Status')
plt.tight_layout()
plt.show()

# Findings:
# (a) Diseased patients have lower total proteins and albumin levels
# (b) Albumin/Globulin ratio is significantly lower in diseased patients
# (c) These findings are consistent with liver dysfunction


# 6. Correlation Between Bilirubin and Enzymes Analysis: Explore relationships between bilirubin levels and liver enzymes.

corr_vars = ['Total_Bilirubin', 'Direct_Bilirubin', 'Alamine_Aminotransferase', 
             'Aspartate_Aminotransferase', 'Liver_Disease']

plt.figure(figsize=(10,8))
sns.heatmap(df[corr_vars].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Between Bilirubin and Liver Enzymes')
plt.show()

# Findings:
# (a) Strong correlation between total and direct bilirubin (0.87)
# (b) Moderate correlation between bilirubin and liver enzymes
# (c) All biomarkers show positive correlation with liver disease


# 7. Age vs. Enzyme Levels Analysis: Examine how enzyme levels vary with age.

plt.figure(figsize=(15,5))
sns.scatterplot(x='Age', y='Alamine_Aminotransferase', hue='Liver_Disease', 
                data=df, alpha=0.6)
plt.title('Age vs ALT Levels by Disease Status')
plt.show()

#Findings:
# (a) Younger patients with disease tend to have extremely high ALT levels
# (b) Older patients generally show more moderate elevation
# (c) Healthy patients maintain low ALT levels regardless of age



######################################## Apply Models ###############################################


# Create features and targets
X = df.drop(['Liver_Disease', 'Total_Bilirubin', 'Direct_Bilirubin'], axis=1)  # Features
y_class = df['Liver_Disease']  # Classification target (1: disease, 0: no disease)
y_reg = df['Total_Bilirubin']  # Regression target

# Split data
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(X, y_class, y_reg, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Classification Models (Support Vector Machine)

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train_scaled, y_class_train)
svm_pred = svm_clf.predict(X_test_scaled)

print("\nSVM Results:")
print(f"Accuracy: {accuracy_score(y_class_test, svm_pred):.2f}")
print("Classification Report:")
print(classification_report(y_class_test, svm_pred))


# Regression Models (Random Forest)

rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train_scaled, y_reg_train)
rf_reg_pred = rf_reg.predict(X_test_scaled)

print("\nRandom Forest Regressor Results:")
print(f"Mean Squared Error: {mean_squared_error(y_reg_test, rf_reg_pred):.2f}")
print(f"R-squared: {r2_score(y_reg_test, rf_reg_pred):.2f}")

# Actual vs Predicted plot (Random Forest)
plt.figure(figsize=(8,6))
plt.scatter(y_reg_test, rf_reg_pred, alpha=0.5)
plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--')
plt.xlabel('Actual Bilirubin')
plt.ylabel('Predicted Bilirubin')
plt.title('Actual vs Predicted Bilirubin Levels (Random Forest)')
plt.show()
