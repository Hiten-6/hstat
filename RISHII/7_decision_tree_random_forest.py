"""
Experiment 7 - Decision Tree and Random Forest
Run this file independently.
Required packages: numpy, pandas, matplotlib, seaborn, scipy, scikit-learn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

print("\nEXPERIMENT 7 — DECISION TREE & RANDOM FOREST")
data = load_iris(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}
scores = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
        "F1-score": f1_score(y_test, pred, average="weighted", zero_division=0)
    }
    scores.append(row)
    print(f"\n{name}")
    for k, v in row.items():
        if k != "Model": print(f"{k}: {v:.4f}")
    print(classification_report(y_test, pred, target_names=data.target_names))
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=data.target_names, yticklabels=data.target_names)
    plt.title(f"{name} — Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); plt.show()
    if name == "Decision Tree":
        plt.figure(figsize=(14, 8))
        plot_tree(model, feature_names=data.feature_names,
                  class_names=data.target_names, filled=True, rounded=True)
        plt.title("Decision Tree"); plt.tight_layout(); plt.show()
    print("Feature importance:")
    print(pd.Series(model.feature_importances_, index=data.feature_names).sort_values(ascending=False))
print("\nMODEL COMPARISON:\n", pd.DataFrame(scores).set_index("Model").round(4))
