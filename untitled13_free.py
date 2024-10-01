import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def prepare_data(ticker, period):
    try:
        data = yf.download(ticker, period=period)
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

st.title('ALGO_PROF')

ticker = st.text_input('Enter a ticker symbol:', 'AAPL')

period_options = ['6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
free_options = ['6mo', '1y']
premium_options = ['2y', '5y', '10y', 'ytd', 'max']

selected_period = st.selectbox(
    'Select data range:',
    period_options,
    format_func=lambda x: f"{x} {'(Premium)' if x in premium_options else ''}"
)

if selected_period in premium_options:
    st.warning("This is a premium feature. Please upgrade to access this data range.")
    st.stop()

if st.button('Predict'):
    st.write(f"Processing {ticker} for {selected_period}...")
    data = prepare_data(ticker, selected_period)
    if data is not None and len(data) > 100:
        accuracy, prediction, feature_importance = run_model(data)
        tickers = yf.Ticker(ticker)
        
        st.subheader(f"Results for {ticker}:")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Prediction: {'Increase' if prediction == 1 else 'Decrease'}")
        st.write(f"Recent Close: ${data['Close'].iloc[-1]:.2f}")
        st.write(f"Recent Date: {data.index[-1].strftime('%Y-%m-%d')}")
        
        st.subheader("Recent Price Chart:")
        st.line_chart(data['Close'])
        
    else:
        st.write("Unable to process this ticker. Please try another or select a different data range.")

st.sidebar.title("About")
st.sidebar.info("This app uses a Machine Learning model to predict whether a stock's price will increase or decrease based on historical data. The model's accuracy is based on past performance and does not guarantee future results. Always do your own research before making investment decisions.")

st.sidebar.title("Disclaimer")
st.sidebar.warning("This app is for educational purposes only. It is not financial advice and should not be used as the basis for any financial decisions.")

st.sidebar.title("Premium Features")
st.sidebar.info("Upgrade to access extended data ranges: 1y, 2y, 5y, 10y, and ytd.")