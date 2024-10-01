# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 17:45:50 2024

@author: Cell Physiology
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df['Symbol'].tolist()
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    return tickers

def prepare_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    try:
        data = yf.download(ticker, period="max")
        data['Returns'] = data['Close'].pct_change()
        data['MA5'] = data['Close'].rolling(window=5).mean()
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['Target'] = (data['Returns'] > 0).astype(int)
        data = data.dropna()
        return data
    except:
        return None

def run_model(data):
    features = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA5', 'MA20']
    X = data[features]
    y = data['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    latest_data = X.iloc[-1].values.reshape(1, -1)
    prediction = rf_model.predict(latest_data)[0]
    feature_importance = dict(zip(features, rf_model.feature_importances_))
    
    return accuracy, prediction, feature_importance

st.title('S&P 500 Stock Prediction App')

if st.button('Run Analysis'):
    with st.spinner('Processing S&P 500 stocks...'):
        tickers = get_sp500_tickers()
        results = []

        progress_bar = st.progress(0)
        for i, ticker in enumerate(tickers):
            st.text(f"Processing {ticker}...")
            data = prepare_data(ticker)
            if data is not None and len(data) > 100:
                accuracy, prediction, feature_importance = run_model(data)
                recent_close = data['Close'].iloc[-1]
                recent_date = data.index[-1].strftime('%Y-%m-%d')
                results.append({
                    'Ticker': ticker,
                    'Accuracy': accuracy,
                    'Prediction': 'Increase' if prediction == 1 else 'Decrease',
                    'Recent_Close': recent_close,
                    'Recent_Date': recent_date,
                    'Close_Importance': feature_importance['Close'],
                    'Open_Importance': feature_importance['Open'],
                    'High_Importance': feature_importance['High'],
                    'Low_Importance': feature_importance['Low'],
                    'Volume_Importance': feature_importance['Volume'],
                    'MA5_Importance': feature_importance['MA5'],
                    'MA20_Importance': feature_importance['MA20']
                })
            progress_bar.progress((i + 1) / len(tickers))

        results_df = pd.DataFrame(results)

        # Sort results: Positive predictions first, then by accuracy
        results_df['Prediction_Numeric'] = (results_df['Prediction'] == 'Increase').astype(int)
        results_df = results_df.sort_values(['Prediction_Numeric', 'Accuracy'], ascending=[False, False])
        results_df = results_df.drop('Prediction_Numeric', axis=1)

    st.success('Analysis complete!')

    st.subheader('Results')
    st.dataframe(results_df)

    # Download button for CSV
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name="sp500_predictions.csv",
        mime="text/csv",
    )

st.sidebar.info('This app analyzes S&P 500 stocks and predicts their future performance using a machine leanring.')
st.sidebar.warning('Please note that this is for educational purposes only and should not be used for actual trading decisions.')