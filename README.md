Objective:
Learn how to clean and prepare raw data for machine learning (ML).

Tools Used:
Python, Pandas, NumPy, Matplotlib, Seaborn

Steps Followed:
1. Loaded the Titanic dataset.
2. Explored data using head(), info(), describe(), and checked nulls.
3. Handled missing values using mean, median, or mode.
4. Encoded categorical variables using Label Encoding and One-Hot Encoding.
5. Scaled features using StandardScaler.
6. Detected outliers using seaborn boxplots.
7. Saved the cleaned dataset.

   
What I Learned:
Data cleaning, handling nulls, encoding techniques, feature scaling, and basic EDA.
Interview Questions & Answers:
1. What are the different types of missing data?
Answer: MCAR, MAR, MNAR
2. How do you handle categorical variables?
Answer: Using Label Encoding or One-Hot Encoding.
3. Difference between normalization and standardization?
Answer: Normalization scales between 0-1. Standardization makes mean 0, std dev 1.
4. How do you detect outliers?
Answer: Using boxplots or IQR method.
5. Why is preprocessing important in ML?
Answer: It cleans the data and improves model performance.
6. One-hot encoding vs label encoding?
Answer: One-hot for nominal data, label encoding for ordinal data.
8. How do you handle data imbalance?
Answer: Using SMOTE, oversampling, or undersampling.
9. Can preprocessing affect model accuracy?
Answer: Yes, it can significantly improve it.



CODE::

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

print(df.head())
print(df.info())
print(df.describe())
print("\nMissing values:\n", df.isnull().sum())

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Cabin'].fillna("Unknown", inplace=True)
df.drop(columns=['Ticket'], inplace=True)

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

plt.figure(figsize=(10, 5))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title('Boxplots for Age and Fare')
plt.show()

df.to_csv("titanic_cleaned.csv", index=False)
print("\nCleaned dataset saved as 'titanic_cleaned.csv'")
