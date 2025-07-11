import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Streamlit app title
st.title("Stock Price and S&P 500 Forecasting App")

# Sidebar for model selection
model_type = st.sidebar.selectbox("Select Model", ["Stock Price (Regression)", "S&P 500 (ARIMA)"])

# --- Stock Price Prediction (Regression) ---
if model_type == "Stock Price (Regression)":
    st.header("Stock Price Prediction (Random Forest)")
    
    try:
        # Load pre-trained regression model
        reg_model = joblib.load('stock_price_regressor.joblib')
        
        # Load stock data for feature reference
        stock_data = pd.read_csv('stock_data.csv')
        
        # Input for prediction
        st.subheader("Enter Features for Prediction")
        lag1 = st.number_input("Lag1 (Previous day close)", value=stock_data['Close'].iloc[-1], format="%.2f")
        lag2 = st.number_input("Lag2 (Two days ago close)", value=stock_data['Close'].iloc[-2], format="%.2f")
        ma5 = st.number_input("MA5 (5-day moving average)", value=stock_data['MA5'].iloc[-1], format="%.2f")
        ma10 = st.number_input("MA10 (10-day moving average)", value=stock_data['MA10'].iloc[-1], format="%.2f")
        volume = st.number_input("Volume", value=float(stock_data['Volume'].iloc[-1]), format="%.0f")
        
        if st.button("Predict Stock Price"):
            # Prepare input data
            input_data = np.array([[lag1, lag2, ma5, ma10, volume]])
            
            # Make prediction
            prediction = reg_model.predict(input_data)[0]
            st.success(f"Predicted Stock Price: ${prediction:.2f}")
            
            # Plot recent stock prices
            plt.figure(figsize=(10, 5))
            plt.plot(stock_data['Date'].tail(30), stock_data['Close'].tail(30), label="Historical Prices")
            plt.axhline(y=prediction, color='r', linestyle='--', label=f"Predicted Price: ${prediction:.2f}")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.title("Recent Stock Prices with Prediction")
            plt.legend()
            plt.xticks(rotation=45)
            st.pyplot(plt)
            
    except FileNotFoundError:
        st.error("Please ensure 'stock_price_regressor.joblib' and 'stock_data.csv' are in the same directory.")

# --- S&P 500 Forecasting (ARIMA) ---
elif model_type == "S&P 500 (ARIMA)":
    st.header("S&P 500 Forecasting (ARIMA)")
    
    try:
        # Load pre-trained ARIMA model
        with open('sp500_arima.pkl', 'rb') as f:
            arima_model_fit = pickle.load(f)
        
        # Load S&P 500 data for visualization
        sp500_data = pd.read_csv('sp500_data.csv')
        sp500_series = sp500_data['Close']
        
        # Input for forecast
        forecast_steps = st.number_input("Number of forecast steps (days)", min_value=1, max_value=30, value=10)
        
        if st.button("Forecast S&P 500"):
            # Generate forecast
            forecast = arima_model_fit.forecast(steps=forecast_steps)
            
            # Create forecast dates
            last_date = pd.to_datetime(sp500_data['Date'].iloc[-1])
            forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_steps)]
            
            # Plot historical data and forecast
            plt.figure(figsize=(10, 5))
            plt.plot(sp500_data['Date'].tail(30), sp500_series.tail(30), label="Historical S&P 500")
            plt.plot(forecast_dates, forecast, color='red', label="Forecast")
            plt.xlabel("Date")
            plt.ylabel("Index Value")
            plt.title("S&P 500 Forecast")
            plt.legend()
            plt.xticks(rotation=45)
            st.pyplot(plt)
            
            # Display forecast values
            st.write("Forecasted Values:", pd.Series(forecast, index=forecast_dates))
            
    except FileNotFoundError:
        st.error("Please ensure 'sp500_arima.pkl' and 'sp500_data.csv' are in the same directory.")

# Instructions
st.write("""
### Instructions
1. **Stock Price (Regression)**: Enter feature values (e.g., lagged prices, moving averages, volume) to predict the next stock price.
2. **S&P 500 (ARIMA)**: Specify the number of days to forecast the S&P 500 index.
3. Ensure the model files ('stock_price_regressor.joblib', 'sp500_arima.pkl') and data files ('stock_data.csv', 'sp500_data.csv') are available.
""")
