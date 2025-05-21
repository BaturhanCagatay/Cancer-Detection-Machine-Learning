import streamlit as st
import pandas as pd
import pickle

# Sayfa başlığı ve ayarı
st.set_page_config(page_title="Meme Kanseri Tahmini", page_icon="🩺", layout="wide")
st.markdown("<h1 style='text-align: center;'>🧬 Meme Kanseri Tanı Uygulaması</h1>", unsafe_allow_html=True)

# MODELLERİ ve SCALER'ı yükle
with open("voting_model.pkl", "rb") as f:
    voting_model = pickle.load(f)
with open("logistic_model.pkl", "rb") as f:
    logistic_model = pickle.load(f)
with open("random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("svc_model.pkl", "rb") as f:
    svc_model = pickle.load(f)
with open("gradient_boosting_model.pkl", "rb") as f:
    gb_model = pickle.load(f)
with open("stacking.pkl", "rb") as f:
    stck_model = pickle.load(f)
with open("SGDC.pkl", "rb") as f:
    sgdc_model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

models = {
    "Voting Classifier": voting_model,
    "Logistic Regression": logistic_model,
    "Random Forest": rf_model,
    "Support Vector Classifier": svc_model,
    "Gradient Boosting": gb_model,
    "Stacking":stck_model,
    "SGDC Classifier":sgdc_model


}

# Doğru sıralı özellik listesi (diagnosis çıkarıldı)
feature_names = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"
]

# Sol panel → Model seçimi
with st.sidebar:
    st.markdown("### 🔍 Model Seçimi")
    selected_model_name = st.selectbox("Tahmin Modeli", list(models.keys()))
    model = models[selected_model_name]

# Ana içerik → Sağlık verileri ve giriş yöntemi
st.markdown("## 🧠 Lütfen Sağlık Bilgilerinizi ve Tümör Özelliklerini Girin")

# Profil seçimi
profile = st.selectbox("🎯 Girdi yöntemi seçin:", ["Manuel Giriş", "Benign Örnek", "Malignant Örnek", "CSV Yükle (.csv)"])
input_data = []

if profile == "Benign Örnek":
    input_data = [
        12.45, 14.3, 82.5, 477.1, 0.1186, 0.088, 0.053, 0.019, 0.157, 0.061,
        0.296, 0.979, 2.029, 23.18, 0.007, 0.014, 0.017, 0.006, 0.019, 0.002,
        14.5, 20.3, 95.5, 600.3, 0.150, 0.120, 0.090, 0.035, 0.215, 0.068
    ]

elif profile == "Malignant Örnek":
    input_data = [
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.0787,
        1.095, 0.9053, 8.589, 153.4, 0.006, 0.049, 0.053, 0.015, 0.030, 0.006,
        25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]

elif profile == "CSV Yükle (.csv)":
    uploaded_file = st.file_uploader("📂 Tek satırlık veri yükleyin:", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Eğer "diagnosis" varsa çıkar
        if "diagnosis" in df.columns:
            df = df.drop(columns=["diagnosis"])

        if df.shape[1] == 30:
            input_data = df.iloc[0].tolist()
        else:
            st.warning("⚠️ CSV dosyası 30 özellik içermeli (diagnosis hariç)")

# Form: Manuel veya otomatik değer gösterimi
if profile == "Manuel Giriş" or not input_data:
    with st.form("kanser_form"):
        gender = st.selectbox("🧍 Cinsiyet", ["Kadın", "Erkek"])
        age = st.slider("📅 Yaş", 1, 100, 30)
        input_data = []
        for feature in feature_names:
            value = st.slider(f"{feature}", 0.0, 30.0, 10.0, step=0.01)
            input_data.append(value)
        submitted = st.form_submit_button("🚀 Tahmin Et")
else:
    # Diğer yöntemlerde form gerekmez
    gender = "Belirtilmedi"
    age = 30
    submitted = st.button("🚀 Tahmin Et")

# Tahmin sonucu
if submitted and input_data:
    try:
        input_df = pd.DataFrame([input_data], columns=feature_names)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        st.markdown("## 📋 Tahmin Sonucu")
        st.write(f"**Yaş:** {age} | **Cinsiyet:** {gender}")
        if prediction == 1:
            st.error("⚠️ Bu tümör **kötü huylu (Malignant)** olabilir.")
        else:
            st.success("✅ Bu tümör **iyi huylu (Benign)** olabilir.")

        st.markdown("### 📊 Güven Skoru")
        st.write(f"**Benign:** {prediction_proba[0]*100:.2f}%")
        st.write(f"**Malignant:** {prediction_proba[1]*100:.2f}%")

    except Exception as e:
        st.error(f"Hata: {e}")
