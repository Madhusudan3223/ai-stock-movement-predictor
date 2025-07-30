import pandas as pd
import ta

# Step 1: Load your raw CSV
df = pd.read_csv("sample_data.csv")

# Step 2: Ensure 'Date' column is datetime
df['Date'] = pd.to_datetime(df['Date'])

# Step 3: Calculate indicators using `ta` library
# Bollinger Bands
bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
df['bollinger_upper'] = bb.bollinger_hband()
df['bollinger_lower'] = bb.bollinger_lband()

# MACD
macd = ta.trend.MACD(close=df['Close'])
df['macd'] = macd.macd()

# RSI
df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()

# Daily Returns
df['returns'] = df['Close'].pct_change()

# Drop rows with NaNs from indicator calculations
df.dropna(inplace=True)

# Save fixed CSV
df.to_csv("fixed_data.csv", index=False)
print("✅ fixed_data.csv saved with all required features.")
