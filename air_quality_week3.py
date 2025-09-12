# air_quality_week3.py
# Final Project Submission - Week 3
# Lakshya Srivastav

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# ---------------------------
# 1. Load dataset
# ---------------------------
print("\n--- Dataset Info ---")
df = pd.read_csv("air_quality.csv")
print(df.info())
print("\nFirst 5 rows:\n", df.head())

# ---------------------------
# 2. Handle missing values
# ---------------------------
numeric_df = df.select_dtypes(include=[np.number])

imputer = SimpleImputer(strategy="median")
df_imputed = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)
# ---------------------------
# 3. Remove outliers (IQR)
# ---------------------------
Q1 = df_imputed.quantile(0.25)
Q3 = df_imputed.quantile(0.75)
IQR = Q3 - Q1
df_clean = df_imputed[~((df_imputed < (Q1 - 1.5 * IQR)) | (df_imputed > (Q3 + 1.5 * IQR))).any(axis=1)]

print(f"\nAfter outlier removal, dataset shape: {df_clean.shape}")

# ---------------------------
# 4. Train-test split & scaling
# ---------------------------
X = df_clean.drop("AQI", axis=1)
y = df_clean["AQI"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---------------------------
# 5. Define models
# ---------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Support Vector Regressor": SVR()
}

results = {}

# ---------------------------
# 6. Train & evaluate models
# ---------------------------
print("\n--- Model Training & Evaluation ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    cv = cross_val_score(model, X_scaled, y, cv=5).mean()

    results[name] = {"MSE": mse, "R2": r2, "CV": cv}

    print(f"\n{name}:")
    print(f"  MSE      : {mse:.2f}")
    print(f"  R2 Score : {r2:.3f}")
    print(f"  CV Score : {cv:.3f}")

    # Save model
    joblib.dump(model, f"{name.lower().replace(' ', '_')}_model.pkl")

# ---------------------------
# 7. Visualizations
# ---------------------------

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df_clean.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

# Feature importance (Random Forest)
rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
features = X.columns

plt.figure(figsize=(8,6))
sns.barplot(x=importances, y=features)
plt.title("Random Forest Feature Importances")
plt.savefig("rf_feature_importances.png")
plt.close()

# Actual vs Predicted (Gradient Boosting)
gb_model = models["Gradient Boosting"]
gb_preds = gb_model.predict(X_test)

plt.figure(figsize=(6,6))
plt.scatter(y_test, gb_preds, alpha=0.5)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Gradient Boosting: Actual vs Predicted AQI")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.savefig("gradient_boosting_actual_vs_pred.png")
plt.close()

print("\nWeek 3 Final Project complete ✅")
