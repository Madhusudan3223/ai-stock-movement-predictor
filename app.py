import streamlit as st
import pandas as pd
import joblib

# -------------------- Page Config --------------------
st.set_page_config(page_title="📈 Market Movement Predictor", layout="centered")

# -------------------- Title & Description --------------------
st.title("📊 NIFTY Market Movement Predictor")
st.markdown("""
This AI model uses technical indicators (like MACD, RSI, Bollinger Bands) to predict whether the market will **go up** (Buy) or **go down** (Sell).
""")

# -------------------- Load Model & Features --------------------
try:
    model = joblib.load("xgb_model.pkl")
    top_features = joblib.load("top_features.pkl")
except Exception as e:
    st.error(f"❌ Error loading model or features: {e}")
    st.stop()

# -------------------- User Input --------------------
st.subheader("📥 Enter Today's Technical Indicator Values")

user_input = {}
for feature in top_features:
    user_input[feature] = st.number_input(f"{feature}", step=0.01, format="%.4f")

# -------------------- Prediction --------------------
if st.button("🔮 Predict Market Movement"):
    try:
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.success("📈 Prediction: Market Likely to Go **Up** (Buy Signal)")
        else:
            st.warning("📉 Prediction: Market Likely to Go **Down** (Sell Signal)")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
