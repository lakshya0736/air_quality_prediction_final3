# Air Quality Prediction - Final Project (Week 3)

This project is part of the **AI/ML Environmental Monitoring & Pollution Control** course.  
It predicts **Air Quality Index (AQI)** based on pollutant levels using machine learning models.

---

## 📊 Dataset
- File: `air_quality.csv`
- Shape: ~29,500 rows × 16 columns
- Columns include pollutants (PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene), AQI, City, Date, and AQI_Bucket.
- For modeling, only numeric columns were used.

---

## ⚙️ Preprocessing
- **Median imputation** for missing values (numeric features only).
- **Outlier removal** using IQR method.
- **Feature scaling** with StandardScaler.

---

## 🤖 Models Trained
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- Support Vector Regressor (SVR)  

All models were evaluated using:
- Mean Squared Error (MSE)  
- R² Score  
- 5-fold Cross-Validation Score  

---

## 📈 Results (Week 3)

| Model                | MSE   | R²    | CV Score |
|----------------------|-------|-------|----------|
| Linear Regression    | 543.7 | 0.699 | 0.600    |
| Decision Tree        | 871.7 | 0.518 | 0.151    |
| Random Forest        | 446.0 | 0.753 | 0.601    |
| Gradient Boosting    | 447.2 | 0.753 | 0.652    |
| Support Vector Regr. | 695.6 | 0.615 | 0.516    |

✅ **Best Models:** Random Forest & Gradient Boosting  

---

## 📊 Visualizations
- Correlation Heatmap (`correlation_heatmap.png`)
- Random Forest Feature Importances (`rf_feature_importances.png`)
- Actual vs Predicted AQI (`actual_vs_pred.png`)

---

## 💾 Saved Models
- `linear_regression_model.pkl`  
- `decision_tree_model.pkl`  
- `random_forest_model.pkl`  
- `gradient_boosting_model.pkl`  
- `support_vector_regressor_model.pkl`  

---

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/lakshya0736/air_quality_prediction_final3.git
   cd air_quality_prediction_final3
