import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load trained XGBoost model
model = pickle.load(open("xgb_model.pkl", "rb"))

# RSI Calculation
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# MACD Calculation
def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, min_periods=1).mean()
    ema_slow = series.ewm(span=slow, min_periods=1).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, min_periods=1).mean()
    return macd, signal_line

# Streamlit UI
st.set_page_config(page_title="📊 Market Movement Predictor", layout="centered")
st.title("📊 AI-Powered Market Movement Predictor")
st.write("Upload your stock/NIFTY data to get up/down predictions based on technical indicators + XGBoost!")

uploaded_file = st.file_uploader("Upload CSV file with columns: Date, Open, High, Low, Close, Volume", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)

        # Technical Indicators
        df['ma7'] = df['Close'].rolling(window=7).mean()
        df['ma21'] = df['Close'].rolling(window=21).mean()
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=7).std()
        df['rsi'] = compute_rsi(df['Close'])
        df['macd'], df['macd_signal'] = compute_macd(df['Close'])
        df['bollinger_upper'] = df['Close'].rolling(20).mean() + 2 * df['Close'].rolling(20).std()
        df['bollinger_lower'] = df['Close'].rolling(20).mean() - 2 * df['Close'].rolling(20).std()

        df.dropna(inplace=True)

        # Live input row
        X_live = df[[
            'ma7', 'ma21', 'returns', 'volatility', 'rsi',
            'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower'
        ]].tail(1)

        # ✅ Fix: Remove spaces in column names to match model
        X_live.columns = X_live.columns.str.strip()

        # Predict
        prediction = model.predict(X_live)[0]
        st.success("📈 Predicted Movement: " + ("📉 Down" if prediction == 0 else "📈 Up"))

    except Exception as e:
        st.error(f"Error processing data: {e}")
