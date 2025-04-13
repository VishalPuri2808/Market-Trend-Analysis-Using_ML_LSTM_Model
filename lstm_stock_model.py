import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def load_sentiment_data():
    sentiment_data = np.random.uniform(-1, 1, size=(len(trainData),))  # Random sentiment scores between -1 and 1
    return sentiment_data

# Load Training Data
data = pd.read_csv('Datasets/Google_train_data.csv')
data["Close"] = pd.to_numeric(data["Close"], errors='coerce')
data = data.dropna()
trainData = data.iloc[:, 4:5].values  # Close Price column



# Get Sentiment Scores
sentiment_scores = load_sentiment_data()

# Normalize Data
scaler = MinMaxScaler(feature_range=(0, 1))
trainData = scaler.fit_transform(trainData)

# Prepare Training Data
X_train, y_train = [], []
for i in range(60, len(trainData)):
    # Reshape trainData[i-60:i, 0] to (60, 1) and sentiment_scores[i-60:i] to (60, 1)
    stock_data_window = trainData[i-60:i, 0].reshape(-1, 1)
    sentiment_data_window = sentiment_scores[i-60:i].reshape(-1, 1)
    
    # Concatenate both arrays along axis=1
    combined_window = np.concatenate((stock_data_window, sentiment_data_window), axis=1)
    
    X_train.append(combined_window)
    y_train.append(trainData[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape X_train to (samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 2))  # 2 features: stock and sentiment


# Define LSTM Model
model = Sequential([
    LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 2)),  # 2 features
    Dropout(0.2),
    LSTM(units=100, return_sequences=True),
    Dropout(0.2),
    LSTM(units=100, return_sequences=True),
    Dropout(0.2),
    LSTM(units=100, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

# Compile the Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=2)

# Save the Model
model.save("lstm_stock_model.h5")  # Save as HDF5 format

# Plot Training Loss
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

print("Model training completed and saved as 'lstm_stock_model.h5'.")
