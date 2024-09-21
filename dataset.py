# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Load the dataset
file_path = '/Australian Vehicle Prices.csv'
df = pd.read_csv(file_path)

# Exploratory Data Analysis (EDA)
# Check for missing values and data types
print(df.info())

# Describe statistical properties of the dataset
print(df.describe())

# Price distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], bins=50, kde=True)
plt.title('Price Distribution')
plt.show()

# Check for non-numeric columns
print(df.dtypes)

# Encode categorical variables before calculating correlation
# Using One-Hot Encoding for categorical columns
df_encoded = pd.get_dummies(df, drop_first=True)

# Correlation heatmap with encoded data
corr_matrix = df_encoded.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Data Preprocessing
# Drop rows where Price is missing
df_encoded = df_encoded.dropna(subset=['Price'])
df_encoded.fillna(df_encoded.median(), inplace=True)

# Feature Correlation with Price
plt.figure(figsize=(10, 6))
sns.heatmap(df_encoded.corr()['Price'].sort_values(ascending=False).to_frame(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation with Price')
plt.show()

# Train-Test Split
X = df_encoded.drop('Price', axis=1)
y = df_encoded['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
print("Linear Regression RMSE:", mean_squared_error(y_test, y_pred_lr, squared=False))
print("Linear Regression R2 Score:", r2_score(y_test, y_pred_lr))

# Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest RMSE:", mean_squared_error(y_test, y_pred_rf, squared=False))
print("Random Forest R2 Score:", r2_score(y_test, y_pred_rf))

# XGBoost Regressor
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost RMSE:", mean_squared_error(y_test, y_pred_xgb, squared=False))
print("XGBoost R2 Score:", r2_score(y_test, y_pred_xgb))

# Hyperparameter Tuning for Random Forest (Optional)
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_
y_pred_tuned_rf = best_rf_model.predict(X_test)
print("Tuned Random Forest RMSE:", mean_squared_error(y_test, y_pred_tuned_rf, squared=False))
print("Tuned Random Forest R2 Score:", r2_score(y_test, y_pred_tuned_rf))
