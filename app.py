import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb

# Load the trained XGBoost model and top features
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('top_features.pkl', 'rb') as f:
    top_features = pickle.load(f)

# Strip feature names just in case
top_features = [f.strip() for f in top_features]

# App title
st.title("📊 AI-Powered Market Movement Predictor")
st.markdown("Upload your stock/NIFTY data to get up/down predictions based on technical indicators + XGBoost!")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file with columns: Date, Open, High, Low, Close, Volume", type=['csv'])

if uploaded_file is not None:
    try:
        # Read and clean the data
        df = pd.read_csv(uploaded_file)

        # Strip column names (fixes mismatch error)
        df.columns = df.columns.str.strip()

        # Show preview
        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

        # Ensure top features exist in data
        if all(f in df.columns for f in top_features):
            # Predict
            X = df[top_features]
            predictions = model.predict(X)

            # Add prediction to dataframe
            df['Prediction'] = ['🔺 Up' if p == 1 else '🔻 Down' for p in predictions]

            # Show results
            st.subheader("📈 Prediction Results")
            st.dataframe(df[['Date'] + top_features + ['Prediction']])
        else:
            missing = [f for f in top_features if f not in df.columns]
            st.error(f"Missing required features in uploaded CSV: {missing}")

    except Exception as e:
        st.error(f"❌ Error processing data: {e}")
