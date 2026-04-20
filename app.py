import streamlit as st
import numpy as np
import plotly.graph_objects as go
import re
import base64

# --- ARKA PLAN AYARI ---
def set_dark_background_css(image_path, brightness=0.2):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        page_bg_css = f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            filter: brightness({brightness});
        }}
        [data-testid="stAppViewContainer"] > .main > div {{ position: relative; z-index: 1; }}
        .stMetric, .stPlotlyChart, h1, h2, h3, h4, p, label {{ color: #f0f2f6 !important; }}
        .stSlider, .stRadio, .stTextInput, .stNumberInput, .stButton {{
            background-color: rgba(30, 34, 42, 0.7);
            color: #f0f2f6 !important;
            border-radius: 8px;
        }}
        .js-plotly-plot .plotly .bg {{ fill: transparent !important; }}
        </style>
        """
        st.markdown(page_bg_css, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"⚠️ Görsel bulunamadı: {image_path}. Dosya adının image_13.PNG olduğundan emin olun.")

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="EKK vs QR Simülasyonu", layout="wide")

if "fonk_metni" not in st.session_state:
    st.session_state.fonk_metni = "np.sin(x) + np.cos(2*x)"

def ekle_metin(ek):
    st.session_state.fonk_metni += ek

# Arka planı image_13.PNG ile kuruyoruz
set_dark_background_css("image_13.PNG", brightness=0.2)

# --- BAŞLIK ---
st.title("📈 Polinom Uydurma: EKK vs QR")
st.markdown("Numerik analiz projesi: Klasik EKK çöküşü ve QR ayrışımı kararlılık testi.")
st.write("---")

# ADIM 1: YÖNTEM
st.subheader("Adım 1: Veri Giriş Yöntemi")
mod = st.radio("Yöntem seçin:", ["🔢 Manuel Noktalar", "📐 Fonksiyon Yaz"], index=None, horizontal=True)

if mod is None:
    st.info("👆 Başlamak için yöntem seçin.")
    st.stop()

# ADIM 2: VERİ GİRİŞİ
if mod == "🔢 Manuel Noktalar":
    st.subheader("Adım 2: Noktaları Ayarla")
    n = st.slider("Nokta Sayısı", 3, 15, 8)
    cols = st.columns(n)
    y_vals = []
    x_verisi = np.arange(n, dtype=float)
    for i in range(n):
        with cols[i]:
            val = st.slider(f"X={i}", -10.0, 20.0, float(np.sin(i)*5+5), 0.5, label_visibility="collapsed")
            y_vals.append(val)
            st.markdown(f"<div style='text-align:center;color:gray;font-size:12px;'>X={i}</div>", unsafe_allow_html=True)
    y_verisi = np.array(y_vals)
else:
    st.subheader("Adım 2: Fonksiyonu Tanımla")
    with st.container(border=True):
        st.text_input("f(x) =", key="fonk_metni")
        b_cols = st.columns(8)
        btns = [("x²", "x^2"), ("x³", "x^3"), ("x⁴", "x^4"), ("+", " + "), ("sin", "np.sin(x)"), ("cos", "np.cos(x)"), ("exp", "np.exp(x)")]
        for idx, (l, v) in enumerate(btns):
            b_cols[idx].button(l, on_click=ekle_metin, args=(v,))
        f_n = st.number_input("Nokta Sayısı", 10, 200, 50)
    x_verisi = np.linspace(-5, 5, f_n)
    raw_f = st.session_state.fonk_metni.replace("^", "**")
    guvenli_f = re.sub(r'(\d)\s*(x|np)', r'\1*\2', raw_f)
    try:
        y_verisi = eval(guvenli_f, {"np": np, "x": x_verisi})
        if isinstance(y_verisi, (int, float)): y_verisi = np.full_like(x_verisi, y_verisi)
    except:
        st.error("⚠️ Geçersiz denklem."); y_verisi = np.zeros(f_n)

# ADIM 3: ANALİZ
st.subheader("Adım 3: Analiz")
max_d = len(x_verisi) - 1 if mod == "🔢 Manuel Noktalar" else 20
derece = st.slider("Polinom Derecesi:", 1, max_d, min(3, max_d))

A = np.vander(x_verisi, N=derece + 1, increasing=True)

# Hesaplamalar
try:
    beta_ekk = np.linalg.inv(A.T @ A) @ A.T @ y_verisi
    mse_ekk = np.mean((y_verisi - (A @ beta_ekk))**2)
    ekk_ok = True
except:
    ekk_ok, mse_ekk = False, float('inf')

Q, R = np.linalg.qr(A)
beta_qr = np.linalg.solve(R, Q.T @ y_verisi)
mse_qr = np.mean((y_verisi - (A @ beta_qr))**2)

# Skorlar
k1, k2, _ = st.columns([1, 1, 2])
k1.markdown(f'<p style="color:#f0f2f6;">EKK Hatası: <b>{mse_ekk:.2e}</b></p>' if ekk_ok else '<p style="color:red;">EKK ÇÖKTÜ</p>', unsafe_allow_html=True)
k2.markdown(f'<p style="color:#f0f2f6;">QR Hatası: <b>{mse_qr:.2e}</b></p>', unsafe_allow_html=True)

# Grafik
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_verisi, y=y_verisi, mode='markers', marker=dict(symbol='diamond', size=8, color='white', line=dict(width=1, color='black')), name='Veri'))
x_c = np.linspace(x_verisi.min()-0.5, x_verisi.max()+0.5, 300)
A_c = np.vander(x_c, N=derece + 1, increasing=True)

if ekk_ok:
    fig.add_trace(go.Scatter(x=x_c, y=A_c @ beta_ekk, mode='lines', line=dict(color='#1f77b4', width=3), name='EKK'))
fig.add_trace(go.Scatter(x=x_c, y=A_c @ beta_qr, mode='lines', line=dict(color='#d62728', width=3, dash='dash'), name='QR'))

fig.update_layout(template="plotly_white", height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#f0f2f6'))
st.plotly_chart(fig, use_container_width=True)
