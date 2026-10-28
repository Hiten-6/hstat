"""
Experiment 5 - Linear Regression and Diagnostics
Run this file independently.
Required packages: numpy, pandas, matplotlib, seaborn, scipy, scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("\nEXPERIMENT 5 — LINEAR REGRESSION & DIAGNOSTICS")
data = load_diabetes(as_frame=True)
X = data.data[["bmi"]]  # single predictor for easy interpretation
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
residuals = y_test - pred
print("Intercept:", round(model.intercept_, 4))
print("Coefficient:", round(model.coef_[0], 4))
print("MAE:", round(mean_absolute_error(y_test, pred), 3))
print("MSE:", round(mean_squared_error(y_test, pred), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 3))
print("R²:", round(r2_score(y_test, pred), 3))
plt.scatter(y_test, pred)
plt.xlabel("Actual target"); plt.ylabel("Predicted target")
plt.title("Actual vs Predicted"); plt.tight_layout(); plt.show()
plt.scatter(pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted target"); plt.ylabel("Residual")
plt.title("Residual Plot"); plt.tight_layout(); plt.show()
