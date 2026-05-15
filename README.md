# Rossmann Store Sales Prediction

Predicting daily sales for 1,115 Rossmann drug stores across Germany using XGBoost.
This project explores how feature engineering and target transformation improve forecast accuracy.

##  Dataset

- **Source:** [Rossmann Store Sales on Kaggle](https://www.kaggle.com/datasets/shahpranshu27/rossman-store-sales)
- **Size:** ~1 million daily sales records across 1,115 stores
- **Target variable:** `Sales` (daily sales per store)

##  Tech Stack

- Python 3
- pandas, numpy
- scikit-learn
- XGBoost
- matplotlib, seaborn
- Jupyter Notebook

##  Approach

1. **Data cleaning** — kept only days where stores were open with sales > 0; filled missing competition and promo info.
2. **Feature engineering** — created date features (year, month, day, weekday, weekend), lag features (7/14/30-day past sales per store), rolling averages and standard deviations, and competition-open duration.
3. **Modeling** — compared Linear Regression baseline against XGBoost; used time-aware train/test split (last 6 weeks as test).
4. **Optimization** — applied log transformation on the target to handle skewed sales distribution.

##  Results

| Model              | MAE    | RMSE    | R²    |
|--------------------|--------|---------|-------|
| Linear Regression  | 957.93 | 1329.89 | 0.810 |
| XGBoost (final)    | 593.86 | 874.01  | 0.918 |

**Final XGBoost model explains ~92% of sales variance**, reducing MAE by 38% over the linear baseline.

##  Key Insights

- **XGBoost massively outperformed linear regression** — non-linear patterns in sales (weekday effects, promotions, seasonality) need a tree-based model.
- **Lag features were critical** — past 7/14/30-day sales captured store-level momentum.
- **Log transformation on the target** handled the right-skewed sales distribution and reduced large errors.
- **Promo flag, day of week, and recent sales trends** are the most important prediction
