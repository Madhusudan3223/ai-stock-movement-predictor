import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volatility import AverageTrueRange
from sklearn.preprocessing import StandardScaler
import joblib

# Page configuration
st.set_page_config(page_title="📊 AI-Powered Market Movement Predictor")

# Title
st.title("📊 AI-Powered Market Movement Predictor")
st.write("Upload your stock/NIFTY data to get up/down predictions based on technical indicators + XGBoost!")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file with columns: Date, Open, High, Low, Close, Volume", type="csv")

def add_features(df):
    df = df.copy()
    df['ma7'] = df['Close'].rolling(window=7).mean()
    df['ma21'] = df['Close'].rolling(window=21).mean()
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=7).std()
    df['rsi'] = RSIIndicator(close=df['Close']).rsi()
    macd = MACD(close=df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    bb = BollingerBands(close=df['Close'])
    df['bollinger_upper'] = bb.bollinger_hband()
    df['bollinger_lower'] = bb.bollinger_lband()
    df = df.dropna()
    return df

def prepare_input(df):
    features = ['ma7', 'ma21', 'returns', 'volatility', 'rsi',
                'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower']
    return df[features]

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # Basic checks
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Missing columns. Required: {', '.join(required_cols)}")
        else:
            df = add_features(df)
            X = prepare_input(df)

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Load model
            model = joblib.load("xgb_model.joblib")

            # Predict
            preds = model.predict(X_scaled)
            df['Prediction'] = np.where(preds == 1, '📈 Up', '📉 Down')

            st.subheader("📋 Predictions Preview")
            st.dataframe(df[['Date', 'Close', 'Prediction']].tail(10))

            # Download option
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions CSV", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error processing data: {e}")
