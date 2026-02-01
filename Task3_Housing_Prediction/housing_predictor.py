import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Data
print("Fetching California Housing data...")
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target  # Target variable (price in $100k)

# 2. Data Cleaning & Feature Selection
# Check for nulls (this dataset is usually clean, but good practice)
if df.isnull().sum().sum() == 0:
    print("Data is clean. No missing values found.")

# We'll use all features for now, but focus on: MedInc, HouseAge, and Rooms
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# 3. Split & Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Model (Random Forest is great for this data)
print("Training the model... please wait.")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Evaluate
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# 6. Save the Model
if not os.path.exists('models'):
    os.makedirs('models')
joblib.dump(model, 'models/housing_model.pkl')
joblib.dump(scaler, 'models/housing_scaler.pkl')
print("\nSuccess: Model and Scaler saved in 'models/' folder.")

# 7. Visualize Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [housing.feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()