import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import requests
import tempfile

# ✅ Setting tampilan
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

# ✅ Mapping pilihan Dataset
dataset_choice = st.sidebar.selectbox("📊 Pilih Dataset:", [
    "BTCUSDT 5m", 
    "BTCUSDT 15m", 
    "ETHUSDT 5m", 
    "ETHUSDT 15m"
])

# ✅ Mapping file Google Drive
csv_links = {
    "BTCUSDT 5m": "https://drive.google.com/uc?export=download&id=1f0bxYrx1zBrFimDERSsCDclkMtkiSvrh",
    "BTCUSDT 15m": "https://drive.google.com/uc?export=download&id=17H7J4JVdOGTWjFMnQfJUGreAy6HCpPRq",
    "ETHUSDT 5m": "https://drive.google.com/uc?export=download&id=1mu4KuwtIFxOVo06neykIl5vQOkoJkk3_",
    "ETHUSDT 15m": "https://drive.google.com/uc?export=download&id=164KjkmNir1J9OF_f4UbyggJE6KVj9_57"
}

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

# ✅ Load CSV dari Google Drive
@st.cache_data
def load_data(url):
    df = pd.read_csv(url, parse_dates=['Datetime'], index_col='Datetime')
    return df

# ✅ Load model keras
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

# ✅ Prediksi 1 langkah ke depan
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

# ✅ Eksekusi prediksi
if dataset_choice:
    df = load_data(csv_links[dataset_choice])

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
