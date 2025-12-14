import streamlit as st
import pandas as pd
import joblib
import os
import gdown

# =============================
# Download MODEL dari Google Drive (karena besar)
# =============================
MODEL_URL = "https://drive.google.com/uc?id=1sK1wrgbOJtuxGXHwBZayAI3427EgL3N-"
MODEL_PATH = "rf_subscription_model.pkl"

if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# =============================
# Load model & scaler
# =============================
model = joblib.load(MODEL_PATH)
scaler = joblib.load("scaler.pkl")  # scaler di GitHub

# =============================
# Konfigurasi halaman
# =============================
st.set_page_config(
    page_title="Prediksi Subscription Pelanggan",
    layout="centered"
)

st.title("📊 Prediksi Subscription Pelanggan")
st.write("Klasifikasi pelanggan menggunakan **Random Forest (Ensemble Method)**")
st.markdown("---")

# =============================
# PILIH MODE
# =============================
mode = st.radio(
    "Pilih Mode Prediksi:",
    ("Input Manual", "Upload CSV")
)

# =============================
# MODE 1: INPUT MANUAL
# =============================
if mode == "Input Manual":
    st.subheader("🧾 Input Data Pelanggan")

    age = st.number_input("Umur", min_value=0, max_value=100, step=1)
    income = st.number_input("Pendapatan", min_value=0.0, step=1000.0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1)
    total_spent = st.number_input("Total Pengeluaran", min_value=0.0, step=1000.0)

    if st.button("🔍 Prediksi"):
        input_df = pd.DataFrame([{
            "age": age,
            "income": income,
            "credit_score": credit_score,
            "total_spent": total_spent
        }])

        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        st.markdown("---")
        if pred == 1:
            st.success(f"✅ BERLANGGANAN (Probabilitas: {prob:.2%})")
        else:
            st.error(f"❌ TIDAK BERLANGGANAN (Probabilitas: {prob:.2%})")

# =============================
# MODE 2: UPLOAD CSV
# =============================
else:
    st.subheader("📂 Upload File CSV")
    st.info("CSV harus memiliki kolom: age, income, credit_score, total_spent")

    file = st.file_uploader(
        "Upload file CSV untuk mulai prediksi",
        type=["csv"]
    )

    if file is not None:
        df = pd.read_csv(file)
        st.write("📄 Data CSV:")
        st.dataframe(df.head())

        df_scaled = scaler.transform(df)
        predictions = model.predict(df_scaled)
        probabilities = model.predict_proba(df_scaled)[:, 1]

        df["prediction"] = predictions
        df["probability"] = probabilities

        st.markdown("### ✅ Hasil Prediksi")
        st.dataframe(df)

        st.download_button(
            "⬇️ Download Hasil Prediksi",
            df.to_csv(index=False),
            file_name="hasil_prediksi_subscription.csv"
        )
