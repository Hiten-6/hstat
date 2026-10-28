"""
Experiment 3 - Sampling
Run this file independently.
Required packages: numpy, pandas, matplotlib, seaborn, scipy, scikit-learn
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

print("\nEXPERIMENT 3 — SAMPLING")
bunch = load_breast_cancer(as_frame=True)
df = bunch.frame.copy()
df["class"] = bunch.target
print("\nOriginal class counts:\n", df["class"].value_counts())
# Random train/test sampling, preserving class proportions
train, test = train_test_split(
    df, test_size=0.25, random_state=42, stratify=df["class"]
)
print("\nStratified train size:", len(train), "\nTest size:", len(test))
print("\nTrain class counts:\n", train["class"].value_counts())
print("\nTest class counts:\n", test["class"].value_counts())
# Demonstrate bootstrap sampling with replacement
boot = resample(train, replace=True, n_samples=len(train), random_state=42)
print("\nBootstrap sample size:", len(boot))
print("Bootstrap class counts:\n", boot["class"].value_counts())
