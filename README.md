# 📦 Sales Orders Prediction with ML

This project explores a large sales dataset to predict the **Number of Orders** using machine learning models. We employ exploratory data analysis, data preprocessing, visualization, feature engineering, and multiple regression techniques, including hyperparameter tuning for model optimization.

---

## 📁 Dataset

- **Name**: `100000 Sales Records.csv`  
- **Size**: 100,000 rows × 14 columns  
- **Source**: Public dataset (unspecified)  
- **Attributes include**:  
  `Region`, `Country`, `Item Type`, `Sales Channel`, `Order Priority`,  
  `Order Date`, `Order ID`, `Ship Date`, `Units Sold`, `Unit Price`,  
  `Unit Cost`, `Total Revenue`, `Total Cost`, `Total Profit`

---

## 🔍 Project Objective

To **predict the number of orders** placed based on categorical and temporal features such as country, item type, sales channel, and order month.

---

## 🧪 Workflow

### 📊 Data Exploration

- Checked for null values and data types
- Converted `Order Date` to datetime and extracted `Order Month`
- Aggregated number of orders grouped by categorical features

### 📈 Visualizations

- Distribution of order counts
- Bar plots of total orders by item type and top 10 countries
- Time series plot of monthly order trends

### ⚙️ Preprocessing

- One-hot encoding on categorical variables
- Standardized features using `StandardScaler`
- Train-test split: 80/20

---

## 🤖 Models Used

| Model                       | MSE     | R²       |
|----------------------------|---------|----------|
| Linear Regression          | 0.1438  | -0.0022  |
| Random Forest Regressor    | 0.1729  | -0.2050  |
| HistGradientBoosting       | 0.1436  | -0.0006  |
| 🔧 Tuned HistGradientBoosting | 0.1435  | -0.0000  |

> ⚠️ **Observation**: Despite tuning, the models performed poorly in terms of R², suggesting that the chosen features do not strongly predict the target variable.

---

## 🛠️ Hyperparameter Tuning

Used `RandomizedSearchCV` on `HistGradientBoostingRegressor`.

Tuned parameters:
- `learning_rate`
- `max_iter`
- `max_depth`
- `min_samples_leaf`
- `l2_regularization`

---

## 📌 Key Insights

- **Feature Engineering**: Added `Order Month` and one-hot encoded categorical variables
- **Feature Importance**: Derived using permutation importance on the tuned HistGradientBoosting model
- **Model Performance**: All models exhibited low R² scores, indicating poor predictive capability from available features alone
- **Next Steps**: Explore additional features (e.g., promotions, stock levels) or alternative prediction targets

---

## 📬 Conclusion

This project serves as a comprehensive case study in:

- 🧱 Handling real-world structured data  
- 🧪 Performing feature engineering  
- 🤖 Applying regression models and tuning them  
- 📊 Interpreting model metrics and visual insights  

> 🚀 Even when models underperform, they teach us _where the data fails_, which is a win in real-world ML.
