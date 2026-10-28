"""
Experiment 2 - Data Preprocessing
Run this file independently.
Required packages: numpy, pandas, matplotlib, seaborn, scipy, scikit-learn
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("\nEXPERIMENT 2 — DATA PREPROCESSING")
# Demonstration dataset with typical numerical/categorical fields and missing values.
df = pd.DataFrame({
    "Age": [20, 21, None, 22, 20],
    "Marks": [78, None, 85, 90, 76],
    "Department": ["AIDS", "AIDS", "COMP", None, "IT"],
    "Result": ["Pass", "Pass", "Pass", "Pass", "Fail"]
})
print("\nOriginal data:\n", df)
num_cols = ["Age", "Marks"]
cat_cols = ["Department"]
df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])
df["Department_Code"] = LabelEncoder().fit_transform(df["Department"])
df[["Age_Scaled", "Marks_Scaled"]] = StandardScaler().fit_transform(df[num_cols])
print("\nAfter imputing missing values, encoding and scaling:\n", df)
print("\nMissing values after preprocessing:\n", df.isnull().sum())
