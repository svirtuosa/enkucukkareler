import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re
import base64

# --- 1. TEMA VE ARKA PLAN ---
def set_bg(image_path):
    try:
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{data}");
            background-size: cover; background-attachment: fixed;
        }}
        [data-testid="stAppViewContainer"]::before {{
            content: ""; position: absolute; top: 0; left: 0; width: 100%; height: 100%;
            background-color: rgba(0, 0, 0, 0.85); /* %85 Karartma */
            z-index: 0;
        }}
        .main .block-container {{ position: relative; z-index: 1; }}
        h1, h2, h3, h4, p, label, .stMetric {{ color: #e0e0e0 !important; }}
        .stButton>button {{ background-color: #2e3136; color: white; border: 1px solid #4f4f4f; }}
        </style>
        """, unsafe_allow_html=True)
    except:
        st.warning("image_13.PNG bulunamadı, standart karanlık tema uygulanıyor.")

st.set_page_config(page_title="EKK vs QR | Analiz Motoru", layout="wide")
set_bg("image_13.PNG")

# --- 2. FONKSİYON YARDIMCILARI ---
if "fonk_metni" not in st.session_state:
    st.session_state.fonk_metni = "x^2 + np.sin(x)"

def ekle_metin(ek):
    st.session_state.fonk_metni += ek

# --- BAŞLIK ---
st.title("📈 EKK vs QR Ayrışımı Analiz Motoru")
st.markdown("Matematik Bölümü Bitirme Projesi | Sayısal Kararlılık Testi")
st.write("---")

# ==========================================
# ADIM 1: YÖNTEM SEÇİMİ
# ==========================================
st.subheader("Adım 1: Veri Kaynağını Belirleyin")
mod = st.radio("Yöntem:", ["🔢 Manuel Noktalar (Slider)", "📐 Fonksiyon ve Hazır Setler"], index=None, horizontal=True)

if mod is None:
    st.info("👆 Lütfen analiz için bir veri giriş yöntemi seçin.")
    st.stop()

x_verisi, y_verisi = np.array([]), np.array([])

# ==========================================
# ADIM 2: VERİ GİRİŞİ
# ==========================================
if mod == "🔢 Manuel Noktalar (Slider)":
    nokta_sayisi = st.slider("Nokta Sayısı", 3, 15, 8)
    cols = st.columns(nokta_sayisi)
    y_list = []
    x_verisi = np.arange(nokta_sayisi, dtype=float)
    for i in range(nokta_sayisi):
        with cols[i]:
            v = st.slider(f"X={i}", -10.0, 20.0, float(np.sin(i)*5+5), step=0.5, label_visibility="collapsed")
            y_list.append(v)
            st.markdown(f"<p style='text-align:center; font-size:12px;'>X={i}</p>", unsafe_allow_html=True)
    y_verisi = np.array(y_list)

else:
    col_f, col_h = st.columns([2, 1])
    with col_h:
        hazir = st.selectbox("Hazır Veri Seti:", ["Seçiniz...", "Ev Fiyatları", "Radar Sinyali", "Sıcaklık"])
        if hazir == "Ev Fiyatları": st.session_state.fonk_metni = "0.5*x + 10"
        elif hazir == "Radar Sinyali": st.session_state.fonk_metni = "np.sin(x) * 5"

    with col_f:
        st.text_input("Denklem f(x) =", key="fonk_metni")
        b_cols = st.columns(8)
        btns = [("x²", "x^2"), ("x³", "x^3"), ("x⁴", "x^4"), ("+", "+"), ("sin", "np.sin(x)"), ("cos", "np.cos(x)"), ("exp", "np.exp(x)"), ("CLR", "clear")]
        for idx, (label, val) in enumerate(btns):
            if label == "CLR":
                if b_cols[idx].button(label): st.session_state.fonk_metni = ""
            else:
                b_cols[idx].button(label, on_click=ekle_metin, args=(val,))
    
    f_n = st.number_input("Nokta Sayısı (Çözünürlük)", 10, 200, 50)
    x_verisi = np.linspace(-5, 5, f_n)
    raw_f = st.session_state.fonk_metni.replace("^", "**")
    guvenli_f = re.sub(r'(\d)\s*(x|np)', r'\1*\2', raw_f)
    try:
        y_verisi = eval(guvenli_f, {"np": np, "x": x_verisi})
    except:
        st.error("⚠️ Denklem hatası."); st.stop()

# ==========================================
# ADIM 3: ANALİZ VE MATRİS MUTFAĞI
# ==========================================
st.write("---")
st.subheader("Adım 3: Polinom Uydurma ve Karşılaştırma")

max_d = len(x_verisi) - 1 if mod == "🔢 Manuel Noktalar (Slider)" else 20
derece = st.slider("Polinom Derecesi (m):", 1, max_d, min(3, max_d))

# --- MATEMATİKSEL MOTOR ---
A = np.vander(x_verisi, N=derece + 1, increasing=True)
AtA = A.T @ A
kappa = np.linalg.cond(AtA)

# EKK Çözümü
try:
    beta_ekk = np.linalg.inv(AtA) @ A.T @ y_verisi
    mse_ekk = np.mean((y_verisi - (A @ beta_ekk))**2)
    ekk_ok = True
except:
    ekk_ok, mse_ekk = False, float('inf')

# QR Çözümü
Q, R = np.linalg.qr(A)
beta_qr = np.linalg.solve(R, Q.T @ y_verisi)
mse_qr = np.mean((y_verisi - (A @ beta_qr))**2)

# --- METRİKLER ---
m1, m2, m3 = st.columns(3)
m1.metric("AᵀA Koşul Sayısı", f"{kappa:.2e}", delta="KRİTİK" if kappa > 1e12 else None, delta_color="inverse")
m2.metric("QR Hatası (MSE)", f"{mse_qr:.2e}")
m3.metric("EKK Hatası (MSE)", f"{mse_ekk:.2e}" if ekk_ok else "ÇÖKTÜ")

if kappa > 1e12:
    st.error(f"🚨 Koşul Sayısı {kappa:.2e} seviyesinde! EKK şu an yuvarlama hataları yapıyor, QR'a güvenin.")

# --- MATRİS MUTFAĞI (USER'S FAVORITE) ---
with st.expander("📂 Matematiksel Mutfak (Matris Görünümleri)"):
    tab1, tab2, tab3, tab4 = st.tabs(["A (Vandermonde)", "AᵀA (Normal Denklem)", "Q (Ortogonal)", "R (Üst Üçgen)"])
    
    def show_matrix(mat, title):
        st.write(f"**{title}** ({mat.shape[0]}x{mat.shape[1]})")
        st.dataframe(pd.DataFrame(mat).style.format("{:.4f}"), use_container_width=True)

    with tab1: show_matrix(A, "Tasarım Matrisi A")
    with tab2: show_matrix(AtA, "Normal Denklem Matrisi AᵀA")
    with tab3: show_matrix(Q, "Ortogonal Matris Q")
    with tab4: show_matrix(R, "Üst Üçgen Matris R")

# --- GRAFİK (PLOTLY) ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_verisi, y=y_verisi, mode='markers', marker=dict(symbol='diamond', size=8, color='white', line=dict(width=1, color='black')), name='Veri'))

x_c = np.linspace(x_verisi.min()-0.5, x_verisi.max()+0.5, 300)
A_c = np.vander(x_c, N=derece + 1, increasing=True)

if ekk_ok:
    fig.add_trace(go.Scatter(x=x_c, y=A_c @ beta_ekk, mode='lines', line=dict(color='cyan', width=2), name='Klasik EKK'))
fig.add_trace(go.Scatter(x=x_c, y=A_c @ beta_qr, mode='lines', line=dict(color='indianred', width=4, dash='dash'), name='QR Ayrışımı'))

fig.update_layout(
    template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    xaxis_title="X Ekseni", yaxis_title="Y Ekseni", height=550,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)
st.plotly_chart(fig, use_container_width=True)

st.caption("🎓 Matematik Bitirme Projesi · EKK vs QR Kararlılık Analizi · ε_mach ≈ 2.22e-16")
