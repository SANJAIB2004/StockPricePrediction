import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st
import matplotlib.pyplot as plt

# Function to load stock data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

# Preprocessing function
def preprocess_data(data):
    data = data[['Date', 'Close']]
    data['Date'] = pd.to_datetime(data['Date'])
    data.index = data['Date']
    data.drop('Date', axis=1, inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Create dataset
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Build LSTM model
def build_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main function for Streamlit app
def main():
    st.title('Stock Price Prediction')
    
    # Date range picker
    start_date = st.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2024-03-01"))
    
    # Sample stock ticker dropdown
    sample_ticker = st.text_input('Enter Stock Ticker', 'AAPL')
    
    if st.button('Predict'):
        data = load_data(sample_ticker, start_date, end_date)
        st.write(data.tail())
        
        # Preprocess data
        scaled_data, scaler = preprocess_data(data)
        
        # Create training and test sets
        training_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[0:training_size, :]
        test_data = scaled_data[training_size:, :]
        
        # Create dataset for LSTM
        time_step = 100
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)
        
        # Reshape input to be [samples, time steps, features]
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Build and train the model
        model = build_model()
        model.fit(X_train, y_train, batch_size=1, epochs=1)
        
        # Predict and reverse scale the data
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        
        # Plot results
        st.subheader('Predicted vs Actual Stock Prices')
        plt.figure(figsize=(14, 7))
        plt.plot(data['Close'], label='Actual Price')
        plt.plot(range(time_step, len(train_predict) + time_step), train_predict, label='Training Prediction')
        plt.plot(range(len(train_predict) + (2 * time_step), len(train_predict) + (2 * time_step) + len(test_predict)), test_predict, label='Test Prediction')
        plt.legend()
        st.pyplot(plt)

if __name__ == '__main__':
    main()
