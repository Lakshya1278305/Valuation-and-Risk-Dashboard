
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Generate synthetic data
np.random.seed(42)
num_samples = 50
X = pd.DataFrame({
    'Revenue': np.random.normal(100, 20, num_samples),
    'Profit_Margin': np.random.uniform(5, 20, num_samples),
    'Debt_to_Equity': np.random.uniform(0.1, 1.0, num_samples),
    'Cash_Flow': np.random.normal(50, 15, num_samples)
})
y = X['Revenue'] * 1.1 + X['Profit_Margin'] * 2 - X['Debt_to_Equity'] * 10 + np.random.normal(0, 5, num_samples)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model_blackrock_forecasting.joblib")

# Evaluate model
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))
