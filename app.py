import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit app
st.set_page_config(page_title="📈 AI Stock Movement Predictor", layout="centered")

st.title("📉 NIFTY Market Movement Predictor")
st.markdown("""
This AI model uses technical indicators (like **MACD**, **RSI**, **Bollinger Bands**) to predict whether the market will **go up** (Buy) or **go down** (Sell).
""")

st.divider()
st.subheader("📊 Enter Today's Technical Indicator Values")

# Input fields
bollinger_upper = st.number_input("📈 Bollinger Upper Band", format="%.4f", step=0.0001)
bollinger_lower = st.number_input("📉 Bollinger Lower Band", format="%.4f", step=0.0001)
macd = st.number_input("🔁 MACD", format="%.4f", step=0.0001)
rsi = st.number_input("🧠 RSI", format="%.4f", step=0.0001)
returns = st.number_input("💹 Daily Returns", format="%.4f", step=0.0001)

# Predict button
if st.button("🧠 Predict Market Movement"):
    input_data = np.array([[bollinger_upper, bollinger_lower, macd, rsi, returns]])
    
    # Model prediction
    prediction = model.predict(input_data)[0]

    # Result display
    if prediction == 1:
        st.success("📈 Market is likely to go **UP** 📈 (Consider Buy)")
    else:
        st.error("📉 Market is likely to go **DOWN** 📉 (Consider Sell)")
