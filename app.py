import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import statsmodels.api as sm
import requests
import tempfile
from datetime import datetime

theme_color = "#ffcc00"
st.set_page_config(page_title="Dashboard Prediksi", layout="wide")

st.markdown(f"""
    <style>
        .sidebar .sidebar-content {{
            background-color: {theme_color};
        }}
        .st-df, .stDataFrameContainer {{
            background-color: #fffaf0;
        }}
    </style>
""", unsafe_allow_html=True)

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/57/Binance_Logo.png", use_container_width=True)
st.sidebar.title("Pengaturan Prediksi")

actual_paths = {
    "Bitcoin 5m": "https://drive.google.com/uc?id=10uGPMZDa9qc6mgGAYbnRWH7H7hWVfeku",
    "Bitcoin 15m": "https://drive.google.com/uc?id=1Fc-HHUmU03TLO8bh_9ODwxujvzdTDqy8",
    "Ethereum 5m": "https://drive.google.com/uc?id=1B2FpsYZ0BHX2Ii2l53QoxpX6KlpqpTDo",
    "Ethereum 15m": "https://drive.google.com/uc?id=1qM4YjlSn5FBomC4AybvyBsqJClT9-g2P"
}

model_paths = {
    "Bitcoin 5m": "https://drive.google.com/uc?id=1DC5CPV8ILlM1X1YVsgQs8QdGuGFijrhh",
    "Bitcoin 15m": "https://drive.google.com/uc?id=1RBf-kUV7mo0Wqx6UmjfC2J5qqDksLvTx",
    "Ethereum 5m": "https://drive.google.com/uc?id=1v9ai1IY44NV-CoslfoY9tN7sdnsw8Ak_",
    "Ethereum 15m": "https://drive.google.com/uc?id=1y1EpHjpFReiEoZc0U0CC71s7fZKicOUF"
}

arima_orders = {
    "Bitcoin 5m": (4, 1, 2),
    "Bitcoin 15m": (0, 1, 2),
    "Ethereum 5m": (4, 1, 4),
    "Ethereum 15m": (4, 1, 3)
}

dataset_choice = st.sidebar.selectbox("Pilih Dataset:", list(actual_paths.keys()))

if dataset_choice:
    freq = "5min" if "5m" in dataset_choice else "15min"

    @st.cache_data
    def load_data(file_url, freq):
        df = pd.read_csv(file_url, parse_dates=['Datetime'], index_col='Datetime')
        df.index = pd.to_datetime(df.index)
        df = df.asfreq(freq).dropna()
        return df
    
    df = load_data(actual_paths[dataset_choice], freq)


    # Pilih tanggal prediksi
    date_choice = st.sidebar.selectbox("Pilih Tanggal Prediksi:", df.index.strftime("%Y-%m-%d %H:%M:%S"))
    predict_button = st.sidebar.button("PREDIKSI")

    # Fungsi buat download model keras
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

    def predict_arima_lstm(df, selected_datetime, lstm_model, arima_order, freq):
        selected_datetime = pd.to_datetime(selected_datetime)
        timestep = df.index.get_loc(selected_datetime)

        if timestep < 4:
            st.error("Data tidak cukup untuk prediksi, pilih tanggal yang lebih besar!")
            return None, None, None, None

        model_arima = sm.tsa.ARIMA(df['Close'].iloc[:timestep], order=arima_order).fit()
        arima_forecast = model_arima.forecast(steps=10)

        selected_features = ['Close']
        input_features = np.tile(df[selected_features].iloc[timestep-4:timestep].values, (10, 1, 1))
        lstm_predictions = lstm_model.predict(input_features).flatten()

        pred_hybrid = arima_forecast[:10] + lstm_predictions
        prediction_dates = pd.date_range(start=selected_datetime, periods=10*3, freq="5min")

        if freq == "15min":
            prediction_dates = [d for d in prediction_dates if d.minute % 15 == 0][:10]
        else:
            prediction_dates = prediction_dates[:10]

        prediction_dates = [d for d in prediction_dates if d in df.index]
        actual_data = df.loc[prediction_dates, 'Close'].values
        selisih = np.abs(np.array(pred_hybrid[:len(actual_data)]) - np.array(actual_data))
        
        return pred_hybrid[:len(actual_data)], prediction_dates, actual_data, selisih

    if predict_button:
        model_path = download_model(model_paths[dataset_choice])
        if model_path:
            lstm_model = tf.keras.models.load_model(model_path)
            predictions, prediction_dates, actual_data, selisih = predict_arima_lstm(df, date_choice, lstm_model, arima_orders[dataset_choice], freq)

            if predictions is not None:
                pred_df = pd.DataFrame({
                    'Datetime': prediction_dates,
                    'Hybrid Prediction': predictions,
                    'Actual Data': actual_data,
                    'Selisih': selisih
                })

                st.subheader("📑 Hasil Prediksi")
                st.dataframe(pred_df.style.set_properties(**{'background-color': '#fffaf0', 'color': 'black'}))

                st.subheader("📈 Grafik Prediksi")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(prediction_dates, predictions, label='Prediksi Hybrid', color='#626F47', linewidth=2)
                ax.plot(prediction_dates, actual_data, label='Data Aktual', color='#A4B465', linestyle='dashed', marker='o')
                ax.set_xlabel("Tanggal", fontsize=12)
                ax.set_ylabel("Harga", fontsize=12)
                ax.set_title(f"Prediksi Harga vs Data Aktual ({dataset_choice})", fontsize=14, fontweight='bold')
                ax.tick_params(axis='x', rotation=45, labelsize=10)
                ax.tick_params(axis='y', labelsize=10)
                ax.grid(True, linestyle='--', alpha=0.6)
                plt.legend(fontsize=10)
                st.pyplot(fig)
