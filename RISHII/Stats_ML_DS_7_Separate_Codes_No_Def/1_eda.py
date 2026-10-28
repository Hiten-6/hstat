"""
Experiment 1 - Exploratory Data Analysis (EDA)
Run this file independently.
Required packages: numpy, pandas, matplotlib, seaborn, scipy, scikit-learn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

print("\nEXPERIMENT 1 — EXPLORATORY DATA ANALYSIS (EDA)")
data = load_iris(as_frame=True)
df = data.frame
df["species"] = df["target"].map(dict(enumerate(data.target_names)))
print("\nFirst 5 rows:\n", df.head())
print("\nShape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nSummary statistics:\n", df.describe())
print("\nMissing values:\n", df.isnull().sum())
print("\nSpecies counts:\n", df["species"].value_counts())
sns.histplot(data=df, x="sepal length (cm)", hue="species", kde=True)
plt.title("Iris — Sepal Length Distribution")
plt.tight_layout(); plt.show()
sns.heatmap(df.select_dtypes("number").corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap"); plt.tight_layout(); plt.show()
sns.pairplot(df, vars=data.feature_names, hue="species")
plt.show()
