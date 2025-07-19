import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import requests

st.title("📈 Prediksi Harga Crypto (Tokocrypto)")

@st.cache_data
def get_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1h", "limit": 500}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", *_])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.astype(float)
    df["target"] = df["close"].shift(-1)
    return df.dropna()

df = get_data()
X = df[["open", "high", "low", "volume"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)
model = LinearRegression()
model.fit(X_train, y_train)

st.write("📊 Data terakhir:")
st.dataframe(df.tail())

pred = model.predict(X.tail(1))
st.success(f"🎯 Prediksi harga close selanjutnya: ${pred[0]:.2f}")
