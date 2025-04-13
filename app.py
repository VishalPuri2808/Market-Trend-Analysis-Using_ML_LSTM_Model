from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests

app = Flask(__name__)

# Load pre-trained LSTM model
try:
    model = load_model("lstm_stock_model.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()
    
# Function to fetch stock news
API_KEY = "80706f87eeec4afc8f3499b8e2b6b88f"
def get_stock_news(ticker):
    try:
        url = f'https://newsapi.org/v2/everything?q={ticker}&language=en&sortBy=publishedAt&apiKey={API_KEY}'
        response = requests.get(url).json()
        
        if 'articles' not in response or not response['articles']:
            print(f"No news found for {ticker}")
            return []
        
        articles = response['articles'][:10]  # Get latest 10 headlines
        news = [article['title'] for article in articles]
        return news
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []

# Function to calculate sentiment score
def analyze_sentiment(news_list):
    if not news_list:
        return 0  # Neutral sentiment if no news is available
    scores = [analyzer.polarity_scores(news)['compound'] for news in news_list]
    return np.mean(scores) if scores else 0  # Average sentiment

# Function to fetch stock data
def get_stock_data(ticker):
    try:
        stock_data = yf.download(ticker, period='2y', interval='1d')
        if stock_data.empty:
            raise ValueError("Stock data is empty.")
        stock_data = stock_data[['Close']]
        return stock_data
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()

# Function to preprocess data
def preprocess_data(data, sentiment_scores=None):
    if data.empty:
        return None, None
    # Scale the data (stock prices)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Add sentiment data if available
    if sentiment_scores is not None:
        sentiment_scores_resized = np.resize(sentiment_scores, (len(scaled_data), 1))
        scaled_data_with_sentiment = np.concatenate((scaled_data, sentiment_scores_resized), axis=1)
        return scaled_data_with_sentiment, scaler
    return scaled_data, scaler

# Function to prepare input data
def prepare_input(data, sentiment_scores=None, timestep=60):
    if len(data) <= timestep:
        return None
    X_test = []
    for i in range(timestep, len(data)):
        stock_data_window = data[i - timestep:i, 0].reshape(-1, 1)
        if sentiment_scores is not None:
            sentiment_data_window = sentiment_scores[i - timestep:i].reshape(-1, 1)
            combined_window = np.concatenate((stock_data_window, sentiment_data_window), axis=1)
        else:
            combined_window = stock_data_window
        X_test.append(combined_window)

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    return X_test

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        ticker = request.form['ticker']
        stock_data = get_stock_data(ticker)
        news = get_stock_news(ticker)
        sentiment_score = analyze_sentiment(news)

        if stock_data.empty:
            return jsonify({'error': 'Stock data unavailable'}), 400

        sentiment_scores = np.array([sentiment_score] * len(stock_data))
        scaled_data, scaler = preprocess_data(stock_data, sentiment_scores)

        if scaled_data is None:
            return jsonify({'error': 'Not enough stock data for prediction'}), 400

        X_test = prepare_input(scaled_data, sentiment_scores, timestep=60)

        if X_test is None:
            return jsonify({'error': 'Not enough historical data for prediction'}), 400

        predictions = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predictions).flatten().tolist()
        actual_prices = stock_data['Close'].values[-len(predicted_prices):].tolist()
        dates = stock_data.index[-len(predicted_prices):].strftime('%m/%d/%Y').tolist()

        return jsonify({
            'predicted_price': round(float(predicted_prices[-1]), 2),
            'actual_price': round(float(stock_data['Close'].values[-1]), 2),
            'sentiment_score': round(float(sentiment_score), 2),
            'news': news,
            'dates': dates,
            'actual': actual_prices,
            'predictions': predicted_prices
        })
    except Exception as e:
        print(f"Error in /predict route: {e}")
        return jsonify({'error': f'Error processing prediction request: {e}'}), 500

@app.route('/predict_trend', methods=['POST'])
def predict_trend():
    ticker = request.form['ticker']
    stock_data = get_stock_data(ticker)

    if stock_data.empty:
        return jsonify({'error': 'Stock data unavailable'}), 400

    scaled_data, scaler = preprocess_data(stock_data)
    if scaled_data is None or len(scaled_data) < 60:
        return jsonify({'error': 'Not enough data for trend prediction'}), 400

    last_60_scaled = scaled_data[-60:]
    X_input = np.array(last_60_scaled).reshape(1, 60, 1)

    predicted_scaled = model.predict(X_input)
    tomorrow_pred = scaler.inverse_transform(predicted_scaled)[0][0]

    today_price = stock_data['Close'].values[-1]

    trend = "UP" if tomorrow_pred > today_price else "DOWN"

    return jsonify({
        'trend': trend, 
        'tomorrow_price': round(float(tomorrow_pred), 2),
        'today_price': round(float(today_price), 2)
    })

@app.route('/stock_info', methods=['POST'])
def stock_info():
    ticker = request.form['ticker']
    stock = yf.Ticker(ticker)
    info = stock.info

    stock_details = {
        "Company Name": info.get("longName", "N/A"),
        "Market Cap": info.get("marketCap", "N/A"),
        "PE Ratio": info.get("trailingPE", "N/A"),
        "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
        "52 Week Low": info.get("fiftyTwoWeekLow", "N/A"),
    }

    return jsonify(stock_details)

if __name__ == '__main__':
    app.run(debug=True)
