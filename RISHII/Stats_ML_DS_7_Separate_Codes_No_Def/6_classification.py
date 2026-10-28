"""
Experiment 6 - Classification (Logistic Regression)
Run this file independently.
Required packages: numpy, pandas, matplotlib, seaborn, scipy, scikit-learn
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

print("\nEXPERIMENT 6 — CLASSIFICATION (LOGISTIC REGRESSION)")
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
model.fit(X_train, y_train)
pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("Precision (weighted):", round(precision_score(y_test, pred, average="weighted"), 4))
print("Recall (weighted):", round(recall_score(y_test, pred, average="weighted"), 4))
print("F1-score (weighted):", round(f1_score(y_test, pred, average="weighted"), 4))
print("\nClassification report:\n", classification_report(y_test, pred, target_names=data.target_names))
cm = confusion_matrix(y_test, pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Confusion Matrix"); plt.tight_layout(); plt.show()
