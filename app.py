import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# ⏳ Load model once
@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")

model = load_model()

# 📈 Technical indicators
def add_technical_indicators(df):
    df['ma7'] = df['Close'].rolling(window=7).mean()
    df['ma21'] = df['Close'].rolling(window=21).mean()
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=7).std()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df['rsi'] = 100 - (100 / (1 + RS))

    df['ema12'] = df['Close'].ewm(span=12).mean()
    df['ema26'] = df['Close'].ewm(span=26).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    df['bollinger_upper'] = df['ma21'] + 2 * df['Close'].rolling(window=21).std()
    df['bollinger_lower'] = df['ma21'] - 2 * df['Close'].rolling(window=21).std()

    return df

# 🧹 Clean and prepare features
def preprocess_data(df):
    df = add_technical_indicators(df)
    df = df.dropna()

    # ✅ Match the feature names and order used in model training
    expected_features = [
        'macd_signal', 'ma7', 'ma21', 'rsi', 'volatility',
        'bollinger_lower', 'returns', 'bollinger_upper', 'macd'
    ]

    # Remove any extra spaces and ensure correct order
    df.columns = [col.strip() for col in df.columns]
    X_live = df[expected_features].copy()

    return X_live, df

# 🚀 Streamlit UI
st.title("📊 AI-Powered Market Movement Predictor")
st.markdown("Upload your stock/NIFTY data to get up/down predictions based on technical indicators + XGBoost!")

uploaded_file = st.file_uploader("Upload CSV file with columns: Date, Open, High, Low, Close, Volume", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # Process data
        X_live, df_processed = preprocess_data(df)

        # Select the most recent row
        latest_data = X_live.tail(1)

        # Make prediction
        prediction = model.predict(latest_data)[0]
        result = "📈 UP" if prediction == 1 else "📉 DOWN"

        st.subheader("Prediction:")
        st.write(f"🔮 The model predicts the market will go: **{result}**")

        # Optional: Show last row of features
        with st.expander("📋 View Latest Feature Data Used"):
            st.dataframe(latest_data)

    except Exception as e:
        st.error(f"❌ Error processing data: {e}")
