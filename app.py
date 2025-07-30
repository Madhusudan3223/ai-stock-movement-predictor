import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

# 📌 Load model and top features
model = joblib.load('xgb_model.pkl')
top_features = joblib.load('top_features.pkl')

# 📌 Functions to calculate indicators
def add_indicators(df):
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=5).std()
    df['ma7'] = df['Close'].rolling(window=7).mean()
    df['ma21'] = df['Close'].rolling(window=21).mean()
    df['rsi'] = compute_rsi(df['Close'])
    df['macd'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['bollinger_upper'] = df['ma21'] + 2 * df['Close'].rolling(21).std()
    df['bollinger_lower'] = df['ma21'] - 2 * df['Close'].rolling(21).std()
    df = df.dropna()
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# 🌐 Streamlit UI
st.set_page_config(page_title="📈 AI Market Movement Predictor", layout="centered")
st.title("📊 AI-Powered Market Movement Predictor")
st.markdown("Upload your stock/NIFTY data to get up/down predictions based on technical indicators + XGBoost!")

# 📤 Upload CSV
uploaded_file = st.file_uploader("Upload CSV file with columns: Date, Open, High, Low, Close, Volume", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    try:
        df = add_indicators(df)
        X_input = df[top_features]
        predictions = model.predict(X_input)
        df['Prediction'] = predictions

        st.subheader("📈 Sample Predictions")
        st.dataframe(df[['Date', 'Close', 'Prediction']].tail(10))

        # 🔵 Visuals
        st.subheader("📉 Close Price with Prediction")
        df_plot = df[['Date', 'Close', 'Prediction']].copy()
        df_plot['Signal'] = df_plot['Prediction'].map({0: 'Down', 1: 'Up'})

        st.line_chart(df_plot.set_index('Date')[['Close']])
        st.bar_chart(df_plot.set_index('Date')['Prediction'])

    except Exception as e:
        st.error(f"Error processing data: {e}")
