# Sales Order Prediction using Machine Learning

This project involves predicting the **Number of Orders** from a large synthetic sales dataset using various regression models, including Linear Regression, Random Forest, and HistGradientBoosting Regressor.

---

## 📁 Dataset

- File: `100000 Sales Records.csv`
- Records: 100,000
- Features: 14 columns including Region, Country, Item Type, Sales Channel, Order Date, Order ID, Units Sold, Revenue, Profit, etc.

---

## 🎯 Objective

To predict the **Number of Orders** per combination of Country, Item Type, Sales Channel, and Month using regression models.

---

## 🧹 Data Preprocessing

- Converted `Order Date` to datetime and extracted the **Order Month**.
- Aggregated data by `Country`, `Item Type`, `Sales Channel`, and `Order Month`.
- One-hot encoded all categorical features.
- Standardized numerical features using `StandardScaler`.

---

## 📊 Exploratory Data Analysis

- Visualized distribution of Number of Orders.
- Analyzed top Item Types and Countries by order volume.
- Tracked monthly order trends.

---

## 🧠 Models Used

1. **Linear Regression**
2. **Random Forest Regressor**
3. **HistGradientBoosting Regressor**
4. **Tuned HistGradientBoosting Regressor** (via `RandomizedSearchCV`)

---

## 🔍 Evaluation Metrics

- **MSE**: Mean Squared Error
- **R²**: Coefficient of Determination

| Model                    | MSE     | R²     |
|-------------------------|---------|--------|
| Linear Regression        | 0.1438  | -0.0022|
| Random Forest            | 0.1729  | -0.2050|
| HistGradientBoosting     | 0.1436  | -0.0006|
| Tuned HistGB             | 0.1435  | -0.0000|

---

## 📌 Key Findings

- The **HistGradientBoosting Regressor** slightly outperformed other models, but the **R² scores remained near zero**, indicating poor predictive power.
- Top influencing features identified using permutation importance:
  - Country
  - Item Type
  - Sales Channel
  - Order Month (certain months had more orders)

---

## ⚠️ Limitations

- All models failed to capture meaningful patterns due to limited variance in `Number of Orders` and heavy class imbalance (mostly 1s).
- One-hot encoding led to **287 features**, which may have diluted the signal in the data.

---

## ✅ Conclusion

While tuning helped marginally, the underlying issue is likely with the **target definition** and lack of strong patterns in the features. Future work should:
- Use continuous targets (e.g., `Units Sold`, `Total Profit`) instead.
- Incorporate additional temporal features.
- Address class imbalance using SMOTE or similar methods.

---

## 🛠️ Tools & Libraries

- Python, Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib

