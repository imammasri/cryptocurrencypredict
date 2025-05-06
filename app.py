import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import tempfile
from datetime import datetime, timedelta
import requests

st.set_page_config(page_title="Dashboard Prediksi Historis", layout="wide")

st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-size: 18px !important;
    }
    .sidebar .sidebar-content {
        background-color: #ffcc00;
    }
    .st-df, .stDataFrameContainer {
        background-color: #fffaf0;
    }
    .css-10trblm {font-size: 28px; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/57/Binance_Logo.png", use_container_width=True)
st.sidebar.title("⚙️ Pengaturan Prediksi")

dataset_choice = st.sidebar.selectbox("📊 Pilih Dataset:", ["BTCUSDT 5m", "BTCUSDT 15m", "ETHUSDT 5m", "ETHUSDT 15m"])

dataset_links = {
    "BTCUSDT 5m": "https://drive.google.com/uc?id=1HbVV1AspJFB79qHpRXFHN-thhcbB8fuJ",
    "BTCUSDT 15m": "https://drive.google.com/uc?id=1qtd0M8v3Aq4p60FclIh86IDIAp6ABrBE",
    "ETHUSDT 5m": "https://drive.google.com/uc?id=1nX8lXVPUhXbnWtWIkYnC5L13Ab8uTIvQ",
    "ETHUSDT 15m": "https://drive.google.com/uc?id=1uznyc9ivPzlTcdBULcrD4M9GEZli-OTX"
}

interval_freq = {
    "5m": "5min",
    "15m": "15min"
}

@st.cache_data

def load_csv_from_drive(link):
    url = link
    df = pd.read_csv(url)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    return df

# Download model dari Drive

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

# Fungsi prediksi 5 langkah ke depan

def predict_multiple_steps(df, lstm_model, arima_order, selected_datetime, interval_label):
    interval_min = int(interval_label.replace("m", ""))
    if selected_datetime not in df.index:
        selected_datetime = df.index[df.index.get_indexer([selected_datetime], method='nearest')[0]]

    idx = df.index.get_loc(selected_datetime)
    if idx < 4:
        idx = 4
        selected_datetime = df.index[idx]

    arima_data = df['Close'].iloc[:idx]
    model_arima = sm.tsa.ARIMA(arima_data, order=arima_order).fit()
    arima_forecast = model_arima.forecast(steps=5)

    input_seq = df['Close'].iloc[idx-4:idx].values
    input_features = np.tile(input_seq, (5, 1)).reshape(5, 4, 1)
    lstm_pred = lstm_model.predict(input_features).flatten()
    hybrid_pred = arima_forecast + lstm_pred

    prediction_dates = pd.date_range(start=df.index[idx], periods=5, freq=interval_freq[interval_label])
    actual_values = df['Close'].reindex(prediction_dates)
    return prediction_dates, hybrid_pred, actual_values

if dataset_choice:
    symbol = dataset_choice.split()[0]
    interval_label = dataset_choice.split()[1]
    df = load_csv_from_drive(dataset_links[dataset_choice])

    if df is not None:
        min_date = df.index.min().date()
        max_date = df.index.max().date()

        selected_date = st.sidebar.date_input("📅 Pilih Tanggal Prediksi", min_value=min_date, max_value=max_date)
        selected_time = st.sidebar.time_input("🕒 Pilih Waktu Prediksi", value=df.index.min().time())
        selected_datetime = pd.to_datetime(f"{selected_date} {selected_time}")

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
            prediction_dates, hybrid_pred, actual_values = predict_multiple_steps(df, lstm_model, arima_orders[dataset_choice], selected_datetime, interval_label)

            pred_df = pd.DataFrame({
                'Datetime': prediction_dates,
                'Hybrid Prediction': hybrid_pred,
                'Actual Data': actual_values.values
            })

            st.subheader("📑 Hasil Prediksi vs Data Aktual")
            st.dataframe(pred_df.style.set_properties(**{'background-color': '#fffaf0', 'color': 'black'}))

            st.subheader("📈 Grafik Prediksi vs Aktual")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(pred_df['Datetime'], pred_df['Hybrid Prediction'], marker='o', linewidth=2, color='#E9A319', label='Hybrid Prediction')
            ax.plot(pred_df['Datetime'], pred_df['Actual Data'], marker='x', linewidth=2, linestyle='--', color='#4C72B0', label='Actual Data')
            ax.set_title(f"Prediksi vs Aktual Harga ({dataset_choice})", fontsize=18, fontweight='bold')
            ax.set_xlabel("Waktu", fontsize=12)
            ax.set_ylabel("Harga (USDT)", fontsize=12)
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            st.pyplot(fig)
