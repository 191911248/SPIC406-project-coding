# Import necessary libraries
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Load market data
data = pd.read_csv('market_data.csv', parse_dates=['Date'], index_col='Date')

# Split data into training and testing sets
train_data = data[:'2022-12-31']
test_data = data['2023-01-01':]

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data)
scaled_test_data = scaler.transform(test_data)

# Define LSTM model architecture
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(scaled_train_data.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile and fit LSTM model to training data
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(scaled_train_data, scaled_train_data, epochs=50, batch_size=32)

# Make predictions on testing data
inputs = scaled_test_data
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predictions = model.predict(inputs)
predictions = scaler.inverse_transform(predictions)

# Evaluate model performance
mse = np.mean((test_data['Close'] - predictions)**2)
mae = np.mean(np.abs(test_data['Close'] - predictions))

# Print model performance metrics
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)