# ✈️ Flight Price Prediction using Machine Learning

This project predicts flight ticket prices using supervised machine learning techniques. The dataset contains over 300,000 flight records with various features such as airline, source/destination cities, stops, travel class, and more. The final model is a Gradient Boosting Regressor that achieved an R² score of **0.97** on the test set.

---

## 🚀 Project Highlights

- **Dataset size:** 300,000+ records
- **Target variable:** `price`
- **Best model:** Gradient Boosting Regressor
- **Test R² Score:** 0.966
- **Cross-Validated R² (Mean):** 0.56

---

## 🧠 Feature Engineering

- Dropped irrelevant features (`flight`, `Unnamed: 0`)
- Encoded categorical variables (`airline`, `class`, `stops`, etc.) using mean encoding and mapping
- Feature scaling was not necessary due to tree-based models
- Transformed time-related features like `departure_time`, `arrival_time` into categorical averages
- Final features include `airline`, `source_city`, `departure_time`, `arrival_time`, `destination_city`, `class`, `duration`, `days_left`, and `stops`

---

## 📊 Exploratory Data Analysis (EDA)

- ✅ Heatmap of correlation matrix for numeric variables
- ✅ Residual analysis to validate model fit
- ✅ Comparison of predictions vs actual prices using scatter plots

---

## 🧪 Model Comparison

| Model                   | R² Score (Cross-Validated Mean) |
|------------------------|-------------------------------|
| Linear Regression       | ~0.03                        |
| Random Forest Regressor | ~0.50                        |
| Gradient Boosting       | **~0.60**                    |
| XGBoost Regressor       | ~0.42                        |

✅ Final Model: **GradientBoostingRegressor**  
✅ Best Params via GridSearchCV:

```python
{
  'learning_rate': 0.1,
  'max_depth': 5,
  'n_estimators': 200,
  'subsample': 0.8
}
