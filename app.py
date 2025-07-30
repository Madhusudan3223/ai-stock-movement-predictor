
---

### ✅ 2. `app.py` (Updated to use `model_xgb.pkl`)

```python
import streamlit as st
import pickle
import numpy as np

# Load the model
with open("model_xgb.pkl", "rb") as f:
    model = pickle.load(f)

# Load top features
with open("top_features.pkl", "rb") as f:
    top_features = pickle.load(f)

# App title and info
st.set_page_config(page_title="NIFTY Predictor", page_icon="📉")
st.title("📉 NIFTY Market Movement Predictor")

st.markdown("""
This AI model uses technical indicators like **MACD**, **RSI**, **Bollinger Bands**, and **Returns** to predict:
- **Buy (1)** if the market may go up 📈
- **Sell (0)** if the market may go down 📉
""")

# Input section
st.header("📊 Enter Today's Technical Indicator Values")

user_input = {}
for feature in top_features:
    user_input[feature] = st.number_input(f"{feature}", value=0.0)

# Prediction logic
if st.button("🔮 Predict Market Movement"):
    input_array = np.array([user_input[f] for f in top_features]).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    
    if prediction == 1:
        st.success("📈 Prediction: BUY (Market likely to go UP)")
    else:
        st.error("📉 Prediction: SELL (Market likely to go DOWN)")
