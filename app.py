from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

app = Flask(__name__)

# Load pre-trained LSTM model
model = tf.keras.models.load_model("lstm_stock_model.h5")  # Load from directory

# Function to fetch stock data
def get_stock_data(ticker):
    stock_data = yf.download(ticker, period='2y', interval='1d')
    stock_data = stock_data[['Close']]
    return stock_data

# Function to preprocess data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Function to prepare input data
def prepare_input(data, scaler):
    timestep = 60
    X_test = []
    for i in range(timestep, len(data)):
        X_test.append(data[i-timestep:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    stock_data = get_stock_data(ticker)

    scaled_data, scaler = preprocess_data(stock_data)
    X_test = prepare_input(scaled_data, scaler)

    predictions = model.predict(X_test)
    
    # Convert predictions and actual prices back to real values
    predicted_prices = scaler.inverse_transform(predictions)
    actual_prices = stock_data['Close'].values[-len(predicted_prices):].reshape(-1, 1)

    # Get corresponding dates
    prediction_dates = stock_data.index[-len(predicted_prices):]

    return jsonify({
        'predictions': predicted_prices.tolist(), 
        'actual': actual_prices.tolist(),
        'dates': prediction_dates.strftime('%m/%d/%Y').tolist()
    })




if __name__ == '__main__':
    app.run(debug=True)
