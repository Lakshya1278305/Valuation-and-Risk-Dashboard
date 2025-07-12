
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define number of samples
num_samples = 100

# Generate synthetic data
data = pd.DataFrame({
    'Revenue': np.random.uniform(50, 500, num_samples).round(2),
    'Profit Margin': np.random.uniform(5, 40, num_samples).round(2),
    'Debt to Equity Ratio': np.random.uniform(0.1, 3.0, num_samples).round(2),
    'Cash Flow': np.random.uniform(10, 200, num_samples).round(2),
})

# Create target variable (simulated next-year revenue)
data['Next Year Revenue'] = (
    data['Revenue'] * (1 + data['Profit Margin'] / 100) 
    - data['Debt to Equity Ratio'] * 5 
    + data['Cash Flow'] * 0.2 
    + np.random.normal(0, 10, num_samples)  # Add some noise
).round(2)

# Save to CSV
data.to_csv("synthetic_financial_data.csv", index=False)
print("Synthetic financial data saved as 'synthetic_financial_data.csv'")
