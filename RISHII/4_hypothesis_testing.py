"""
Experiment 4 - Hypothesis Testing
Run this file independently.
Required packages: numpy, pandas, matplotlib, seaborn, scipy, scikit-learn
"""

from sklearn.datasets import load_iris
from scipy.stats import ttest_ind

print("\nEXPERIMENT 4 — HYPOTHESIS TESTING")
iris = load_iris(as_frame=True).frame
# Compare sepal length of Setosa vs Versicolor (first 50, next 50 rows)
setosa = iris.iloc[:50]["sepal length (cm)"]
versicolor = iris.iloc[50:100]["sepal length (cm)"]
t_stat, p_value = ttest_ind(setosa, versicolor, equal_var=False)
alpha = 0.05
print("H0: Mean sepal length is equal for Setosa and Versicolor.")
print("H1: The means are different.")
print(f"Setosa mean: {setosa.mean():.3f}")
print(f"Versicolor mean: {versicolor.mean():.3f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")
print("Decision:", "Reject H0 (significant difference)" if p_value < alpha
      else "Fail to reject H0 (insufficient evidence)")
