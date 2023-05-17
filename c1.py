# Import necessary libraries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA

# Load stock market data
data = pd.read_csv('stock_data.csv', parse_dates=['Date'], index_col='Date')

# Split data into training and testing sets
train_data = data[:'2022-12-31']
test_data = data['2023-01-01':]

# Define ARIMA model parameters
p = 3  # Autoregressive order
d = 1  # Integrated order
q = 2  # Moving average order

# Fit ARIMA model to training data
model = ARIMA(train_data, order=(p, d, q))
fit_model = model.fit()

# Make predictions on testing data
predictions = fit_model.forecast(steps=len(test_data))

# Evaluate model performance
mse = np.mean((test_data['Close'] - predictions)**2)
mae = np.mean(np.abs(test_data['Close'] - predictions))

# Print model performance metrics
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)