import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Function to prepare data
def prepare_data(ticker):
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

# Function to run the model
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

# Streamlit app
st.title('ALGO_PROF')

# User input
ticker = st.text_input('Enter a ticker symbol:', 'AAPL')

if st.button('Predict'):
    st.write(f"Processing {ticker}...")
    data = prepare_data(ticker)
    if data is not None and len(data) > 100:
        accuracy, prediction, feature_importance = run_model(data)
        tickers = yf.Ticker(ticker)
        
        # Get company info
        try:
            company_name = tickers.info['longName']
            sector = tickers.info['sector']
        except:
            company_name = "N/A"
            sector = "N/A"
        
        # Display results
        # st.subheader(f"Results for {company_name} ({ticker}):")
        # st.write(f"Sector: {sector}")
        st.subheader(f"Results for {ticker}:")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Prediction: {'Increase' if prediction == 1 else 'Decrease'}")
        st.write(f"Recent Close: ${data['Close'].iloc[-1]:.2f}")
        st.write(f"Recent Date: {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Display feature importance
        # st.subheader("Feature Importance:")
        # feature_df = pd.DataFrame.from_dict(feature_importance, orient='index', columns=['Importance'])
        # feature_df = feature_df.sort_values('Importance', ascending=False)
        # st.bar_chart(feature_df)
        
        # Display recent price chart
        st.subheader("Recent Price Chart:")
        st.line_chart(data['Close'].tail(365))
        
    else:
        st.write("Unable to process this ticker. Please try another.")

# Add some information about the app
st.sidebar.title("About")
st.sidebar.info("This app uses a Machine Learning model to predict whether a stock's price will increase or decrease based on historical data. The model's accuracy is based on past performance and does not guarantee future results. Always do your own research before making investment decisions.")

# Add a disclaimer
st.sidebar.title("Disclaimer")
st.sidebar.warning("This app is for educational purposes only. It is not financial advice and should not be used as the basis for any financial decisions.")