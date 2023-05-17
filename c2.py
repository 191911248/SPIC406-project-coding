# Import necessary libraries
import pandas as pd
from fbprophet import Prophet

# Load market data
data = pd.read_csv('market_data.csv')

# Rename columns to fit Prophet's input format
data = data.rename(columns={'Date': 'ds', 'Close': 'y'})

# Create and fit Prophet model
model = Prophet()
model.fit(data)

# Create future dates for prediction
future_dates = model.make_future_dataframe(periods=365)

# Make predictions on future dates
predictions = model.predict(future_dates)

# Plot predictions
model.plot(predictions)

# Plot components of the forecast
model.plot_components(predictions)