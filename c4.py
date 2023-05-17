# Import necessary libraries
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
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

# Reshape data for CNN input
n_features = 1
n_steps = 30
train_X, train_y = [], []
for i in range(n_steps, len(scaled_train_data)):
    train_X.append(scaled_train_data[i-n_steps:i, 0])
    train_y.append(scaled_train_data[i, 0])
train_X, train_y = np.array(train_X), np.array(train_y)
train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], n_features))

test_X, test_y = [], []
for i in range(n_steps, len(scaled_test_data)):
    test_X.append(scaled_test_data[i-n_steps:i, 0])
    test_y.append(scaled_test_data[i, 0])
test_X, test_y = np.array(test_X), np.array(test_y)
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], n_features))

# Define CNN model architecture
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=1))

# Compile and fit CNN model to training data
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_X, train_y, epochs=50, batch_size=32)

# Make predictions on testing data
predictions = model.predict(test_X)
predictions = scaler.inverse_transform(predictions)

# Evaluate model performance
mse = np.mean((test_data['Close'] - predictions)**2)
mae = np.mean(np.abs(test_data['Close'] - predictions))

# Print model performance metrics
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)