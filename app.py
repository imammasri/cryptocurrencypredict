import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import requests
import tempfile

theme_color = "#ffcc00"
st.set_page_config(page_title="Dashboard Prediksi Real-Time", layout="wide")

st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-size: 20px !important;
    }
    .sidebar .sidebar-content {
        background-color: #ffcc00;
    }
    .st-df, .stDataFrameContainer {
        background-color: #fffaf0;
    }
    .css-10trblm {font-size: 28px; font-weight: bold;} /* Header title */
    </style>
""", unsafe_allow_html=True)

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/57/Binance_Logo.png", use_container_width=True)
st.sidebar.title("Pengaturan Prediksi")

dataset_choice = st.sidebar.selectbox("📊 Pilih Dataset:", ["BTCUSDT 5m", "BTCUSDT 15m", "ETHUSDT 5m", "ETHUSDT 15m"])

interval_map = {
    "5m": "5m",
    "15m": "15m"
}

@st.cache_data
def fetch_binance_data(symbol, interval, limit=500):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                                         'Close Time', 'Quote Asset Volume', 'Number of Trades',
                                         'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])
        df['Datetime'] = pd.to_datetime(df['Open Time'], unit='ms')
        df.set_index('Datetime', inplace=True)
        df = df[['Close']].astype(float)
        df.index = df.index + pd.Timedelta(hours=8)  # Konversi waktu ke WITA
        return df
    else:
        st.error("❌ Gagal fetch data dari Binance!")
        return None

def download_model(model_url):
    response = requests.get(model_url)
    if response.status_code == 200:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".keras")
        with open(temp_file.name, "wb") as f:
            f.write(response.content)
        return temp_file.name
    else:
        st.error("❌ Gagal download model!")
        return None

def predict_one_step(df, lstm_model, arima_order):
    latest_data = df[['Close']].iloc[-4:]

    model_arima = sm.tsa.ARIMA(df['Close'], order=arima_order).fit()
    arima_forecast = model_arima.forecast(steps=1)[0]

    input_features = np.expand_dims(latest_data.values, axis=0)
    lstm_prediction = lstm_model.predict(input_features).flatten()[0]

    hybrid_prediction = arima_forecast + lstm_prediction
    actual_price = df['Close'].iloc[-1]
    last_datetime = df.index[-1]

    return last_datetime, hybrid_prediction, actual_price

if dataset_choice:
    symbol = dataset_choice.split()[0]
    interval = interval_map[dataset_choice.split()[1]]

    df = fetch_binance_data(symbol, interval)

    if df is not None:
        model_paths = {
            "BTCUSDT 5m": "https://drive.google.com/uc?id=1DC5CPV8ILlM1X1YVsgQs8QdGuGFijrhh",
            "BTCUSDT 15m": "https://drive.google.com/uc?id=1RBf-kUV7mo0Wqx6UmjfC2J5qqDksLvTx",
            "ETHUSDT 5m": "https://drive.google.com/uc?id=1v9ai1IY44NV-CoslfoY9tN7sdnsw8Ak_",
            "ETHUSDT 15m": "https://drive.google.com/uc?id=1y1EpHjpFReiEoZc0U0CC71s7fZKicOUF"
        }

        arima_orders = {
            "BTCUSDT 5m": (4, 1, 2),
            "BTCUSDT 15m": (0, 1, 2),
            "ETHUSDT 5m": (4, 1, 4),
            "ETHUSDT 15m": (4, 1, 3)
        }

        model_path = download_model(model_paths[dataset_choice])
        if model_path:
            lstm_model = tf.keras.models.load_model(model_path)
            last_datetime, hybrid_prediction, actual_price = predict_one_step(df, lstm_model, arima_orders[dataset_choice])

            selisih = abs(hybrid_prediction - actual_price)

            pred_df = pd.DataFrame({
                'Datetime (WITA)': [last_datetime],
                'Hybrid Prediction': [hybrid_prediction],
                'Actual Data': [actual_price],
                'Selisih': [selisih]
            })

            st.subheader("📑 Hasil Prediksi Real-Time")
            st.dataframe(pred_df.style.set_properties(**{'background-color': '#fffaf0', 'color': 'black'}))

            st.subheader("📈 Grafik Prediksi vs Aktual (WITA)")
            fig, ax = plt.subplots(figsize=(8, 4))

            labels = ['Prediksi Hybrid', 'Data Aktual']
            values = [hybrid_prediction, actual_price]
            colors = ['#A86523', '#E9A319']

            bars = ax.bar(labels, values, color=colors, width=0.4, edgecolor='black', linewidth=1.5)

            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height / 2), 
                            xytext=(0, 10),
                            textcoords="offset points",
                            ha='center', va='center',
                            color='white',fontsize=14, fontweight='bold')

            ax.set_ylabel("Harga (USDT)", fontsize=14, fontweight='bold')
            ax.set_title(f"Perbandingan Prediksi dan Aktual {dataset_choice} (WITA)", fontsize=20, fontweight='bold')
            ax.tick_params(axis='x', labelsize=14)
            ax.tick_params(axis='y', labelsize=14)
            ax.grid(False)
            st.pyplot(fig)
