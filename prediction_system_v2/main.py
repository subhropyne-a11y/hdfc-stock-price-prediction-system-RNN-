from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import pickle
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# Global variables
model = None
scaler = None
df = None
model_metrics = {}

def load_and_train_model():
    """Load data, train model, and save metrics"""
    global model, scaler, df, model_metrics
    
    # Load dataset
    df = pd.read_csv('HDFCBANK.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.sort_values('Date')
    
    # Clean column names
    df.columns = [c.strip().replace(' ', '_').replace('%', 'Pct') for c in df.columns]
    
    # Prepare data for Linear Regression
    data = df[['Close']].copy()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences (X = current price, y = next price)
    X_lr = data_scaled[:-1]
    y_lr = data_scaled[1:]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_lr, y_lr, test_size=0.2, shuffle=False
    )
    
    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test)
    
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    accuracy = 100 - mape
    
    model_metrics = {
        'mse': float(round(mse, 4)),
        'rmse': float(round(rmse, 4)),
        'mape': float(round(mape, 4)),
        'accuracy': float(round(accuracy, 4))
    }
    
    # Save model, scaler, and metrics
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('metrics.pkl', 'wb') as f:
        pickle.dump(model_metrics, f)
    
    print("Model trained successfully!")
    print(f"Model Metrics: {model_metrics}")
    return model_metrics

def load_model():
    """Load pre-trained model and scaler"""
    global model, scaler, df, model_metrics
    
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Load metrics if they exist
        if os.path.exists('metrics.pkl'):
            with open('metrics.pkl', 'rb') as f:
                model_metrics = pickle.load(f)
        else:
            # If metrics don't exist, set defaults
            model_metrics = {
                'mse': 0.0,
                'rmse': 0.0,
                'mape': 0.0,
                'accuracy': 0.0
            }
        
        # Load dataset for historical data
        df = pd.read_csv('HDFCBANK.csv')
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        df = df.sort_values('Date')
        df.columns = [c.strip().replace(' ', '_').replace('%', 'Pct') for c in df.columns]
        
        print("Model loaded successfully!")
        print(f"Loaded Metrics: {model_metrics}")
    except FileNotFoundError:
        print("Model not found. Training new model...")
        model_metrics = load_and_train_model()

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route('/api/historical-data')
def get_historical_data():
    """Return historical stock data"""
    if df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    # Get last 365 days of data
    recent_df = df.tail(365).copy()
    
    data = {
        'dates': recent_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
        'close': recent_df['Close'].tolist(),
        'open': recent_df['Open'].tolist(),
        'high': recent_df['High'].tolist(),
        'low': recent_df['Low'].tolist(),
        'volume': recent_df['Volume'].tolist()
    }
    
    return jsonify(data)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict future stock price"""
    try:
        data = request.json
        current_price = float(data.get('current_price', df['Close'].iloc[-1]))
        days_ahead = int(data.get('days_ahead', 1))
        
        # Scale current price
        current_scaled = scaler.transform([[current_price]])
        
        predictions = []
        dates = []
        last_date = df['Date'].iloc[-1]
        
        # Predict for multiple days
        for i in range(days_ahead):
            pred_scaled = model.predict(current_scaled)
            pred_price = scaler.inverse_transform(pred_scaled)[0][0]
            
            predictions.append(round(pred_price, 2))
            next_date = last_date + timedelta(days=i+1)
            dates.append(next_date.strftime('%Y-%m-%d'))
            
            # Use prediction as input for next day
            current_scaled = pred_scaled
        
        return jsonify({
            'predictions': predictions,
            'dates': dates,
            'current_price': current_price
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/metrics')
def get_metrics():
    """Return model performance metrics"""
    global model_metrics
    
    # Ensure metrics exist
    if not model_metrics or all(v == 0 for v in model_metrics.values()):
        # Recalculate metrics if they don't exist
        try:
            load_and_train_model()
        except Exception as e:
            return jsonify({
                'mse': 0.0,
                'rmse': 0.0,
                'mape': 0.0,
                'accuracy': 0.0,
                'error': str(e)
            }), 500
    
    return jsonify(model_metrics)

@app.route('/api/statistics')
def get_statistics():
    """Return statistical analysis of stock data"""
    if df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    # Calculate statistics
    recent_df = df.tail(365)
    
    stats = {
        'mean': round(recent_df['Close'].mean(), 2),
        'median': round(recent_df['Close'].median(), 2),
        'std': round(recent_df['Close'].std(), 2),
        'min': round(recent_df['Close'].min(), 2),
        'max': round(recent_df['Close'].max(), 2),
        'current': round(df['Close'].iloc[-1], 2),
        'change_1d': round(df['Close'].iloc[-1] - df['Close'].iloc[-2], 2),
        'change_pct': round(((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100, 2)
    }
    
    return jsonify(stats)

@app.route('/api/moving-averages')
def get_moving_averages():
    """Return moving averages data"""
    if df is None:
        return jsonify({'error': 'Data not loaded'}), 500
    
    recent_df = df.tail(365).copy()
    recent_df['MA_50'] = recent_df['Close'].rolling(window=50).mean()
    recent_df['MA_200'] = recent_df['Close'].rolling(window=200).mean()
    
    data = {
        'dates': recent_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
        'close': recent_df['Close'].tolist(),
        'ma_50': recent_df['MA_50'].fillna(0).tolist(),
        'ma_200': recent_df['MA_200'].fillna(0).tolist()
    }
    
    return jsonify(data)

if __name__ == '__main__':
    # Load or train model on startup
    load_model()
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)