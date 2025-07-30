import streamlit as st
import pickle
import numpy as np

# Load trained model
with open('model_xgb.pkl', 'rb') as f:
    model = pickle.load(f)

# Page config
st.set_page_config(page_title="📊 AI-Powered NIFTY Predictor", page_icon="📈", layout="centered")

# Main title
st.markdown("""
    <h1 style='text-align: center;'>📉 NIFTY Market Movement Predictor</h1>
    <p style='text-align: center;'>This AI model uses technical indicators (like MACD, RSI, Bollinger Bands) to predict whether the market will <b>go up</b> (Buy) or <b>go down</b> (Sell).</p>
    <hr>
""", unsafe_allow_html=True)

st.subheader("📉 Enter Today's Technical Indicator Values")

# Input fields
bollinger_upper = st.number_input("bollinger_upper", value=0.0, step=0.0001, format="%.4f")
bollinger_lower = st.number_input("bollinger_lower", value=0.0, step=0.0001, format="%.4f")
macd = st.number_input("macd", value=0.0, step=0.0001, format="%.4f")
rsi = st.number_input("rsi", value=0.0, step=0.0001, format="%.4f")
returns = st.number_input("returns", value=0.0, step=0.0001, format="%.4f")

# Prediction
if st.button("🔮 Predict Market Movement"):
    input_data = np.array([[bollinger_upper, bollinger_lower, macd, rsi, returns]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("📈 Prediction: Market will go UP (Buy signal)")
    else:
        st.error("📉 Prediction: Market will go DOWN (Sell signal)")

# Optional: Add footer or GitHub repo link
st.markdown("""
    <hr>
    <p style='text-align: center;'>Made with ❤️ using Streamlit | <a href='https://github.com/yourusername/yourrepo' target='_blank'>GitHub Repo</a></p>
""", unsafe_allow_html=True)
