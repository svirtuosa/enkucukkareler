import streamlit as st
import numpy as np
import plotly.graph_objects as go
import re

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="EKK vs QR Simülasyonu", layout="wide")

# Fonksiyon girişi için buton hafızasını hazırlıyoruz
if "fonk_metni" not in st.session_state:
    st.session_state.fonk_metni = "np.sin(x) + np.cos(2*x)"

def ekle_metin(ek):
    st.session_state.fonk_metni += ek

# --- BAŞLIK VE GİRİŞ ---
st.title("📈 Polinom Uydurma: EKK vs QR")
st.markdown("Bu interaktif laboratuvar, Klasik EKK'nın yüksek derecelerdeki çöküşünü ve QR ayrışımının kararlılığını test etmenizi sağlar.")
st.write("---")

# ==========================================
# ADIM 1: YÖNTEM SEÇİMİ
# ==========================================
st.subheader("Adım 1: Veri Giriş Yöntemi")
mod = st.radio(
    "Lütfen verilerinizi nasıl oluşturmak istediğinizi seçin:", 
    ["🔢 Noktaları Manuel Belirle (Barlar ile)", "📐 Matematiksel Fonksiyon Yaz"],
    index=None,
    horizontal=True
)

if mod is None:
    st.info("👆 Başlamak için yukarıdan bir yöntem seçin.")
    st.stop()

st.write("---")

x_verisi = np.array([])
y_verisi = np.array([])

# ==========================================
# ADIM 2: VERİ GİRİŞ EKRANI
# ==========================================
if mod == "🔢 Noktaları Manuel Belirle (Barlar ile)":
    st.subheader("Adım 2: Noktaları Ayarla")
    nokta_sayisi = st.slider("Kaç Adet Nokta Olsun?", min_value=3, max_value=15, value=8)
    
    kolonlar = st.columns(nokta_sayisi)
    y_verisi_list = []
    x_verisi = np.arange(nokta_sayisi, dtype=float)
    
    for i in range(nokta_sayisi):
        with kolonlar[i]:
            varsayilan = float(np.sin(i) * 5 + 5)
            val = st.slider(f"X={i}", -10.0, 20.0, varsayilan, 0.5, label_visibility="collapsed")
            y_verisi_list.append(val)
            st.markdown(f"<div style='text-align: center; color: gray; font-size:12px; font-weight:bold;'>X={i}</div>", unsafe_allow_html=True)
    y_verisi = np.array(y_verisi_list)

elif mod == "📐 Matematiksel Fonksiyon Yaz":
    st.subheader("Adım 2: Fonksiyonu Tanımla")
    with st.container(border=True):
        st.text_input("f(x) =", key="fonk_metni")
        st.caption("Hızlı Ekleme Butonları:")
        b_cols = st.columns(8)
        semboller = [("x²", "x^2"), ("x³", "x^3"), ("x⁴", "x^4"), ("x⁵", "x^5"), ("+", " + "), ("sin(x)", "np.sin(x)"), ("cos(x)", "np.cos(x)"), ("exp(x)", "np.exp(x)")]
        
        for idx, (label, val) in enumerate(semboller):
            b_cols[idx].button(label, on_click=ekle_metin, args=(val,))
        
        f_nokta = st.number_input("Çizilecek Nokta Sayısı", 10, 200, 50)
            
    x_verisi = np.linspace(-5, 5, f_nokta)
    guvenli_fonksiyon = st.session_state.fonk_metni.replace("^", "**")
    guvenli_fonksiyon = re.sub(r'(\d)\s*(x|np)', r'\1*\2', guvenli_fonksiyon)
    
    try:
        y_verisi = eval(guvenli_fonksiyon, {"np": np, "x": x_verisi})
        if isinstance(y_verisi, (int, float)): y_verisi = np.full_like(x_verisi, y_verisi)
    except Exception:
        st.error("⚠️ Geçersiz fonksiyon denklemi.")
        y_verisi = np.zeros(f_nokta)

st.write("---")

# ==========================================
# ADIM 3: ANALİZ VE GRAFİK
# ==========================================
st.subheader("Adım 3: Analiz ve Karşılaştırma")

# Dinamik Derece Ayarı
max_d = len(x_verisi) - 1 if mod == "🔢 Noktaları Manuel Belirle (Barlar ile)" else 20
derece = st.slider("Uydurulacak Polinom Derecesi:", 1, max_d, min(3, max_d))

if len(x_verisi) <= derece:
    st.warning("⚠️ Daha fazla nokta veya daha düşük derece seçin.")
    st.stop()

# --- HESAPLAMALAR ---
A = np.vander(x_verisi, N=derece + 1, increasing=True)

# EKK
try:
    beta_ekk = np.linalg.inv(A.T @ A) @ A.T @ y_verisi
    mse_ekk = np.mean((y_verisi - (A @ beta_ekk))**2)
    ekk_basarili = True
except:
    ekk_basarili, mse_ekk = False, float('inf')

# QR
Q, R = np.linalg.qr(A)
beta_qr = np.linalg.solve(R, Q.T @ y_verisi)
mse_qr = np.mean((y_verisi - (A @ beta_qr))**2)

# --- SKOR KARTLARI ---
k1, k2, _ = st.columns([1, 1, 2])
k1.metric("Klasik EKK Hatası (MSE)", f"{mse_ekk:.2e}" if ekk_basarili else "ÇÖKTÜ")
k2.metric("QR Ayrışımı Hatası (MSE)", f"{mse_qr:.2e}")

# --- GRAFİK ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_verisi, y=y_verisi, mode='markers', marker=dict(symbol='diamond', size=8, color='white', line=dict(width=1.5, color='black')), name='Veri'))
x_c = np.linspace(x_verisi.min()-0.5, x_verisi.max()+0.5, 300)
A_c = np.vander(x_c, N=derece + 1, increasing=True)

if ekk_basarili:
    fig.add_trace(go.Scatter(x=x_c, y=A_c @ beta_ekk, mode='lines', line=dict(color='blue', width=4), name='EKK'))
fig.add_trace(go.Scatter(x=x_c, y=A_c @ beta_qr, mode='lines', line=dict(color='indianred', width=4, dash='dash'), name='QR'))

f = max(y_verisi.max() - y_verisi.min(), 1)
fig.update_layout(yaxis_range=[y_verisi.min()-f*0.3, y_verisi.max()+f*0.3], template="plotly_white", height=500)
st.plotly_chart(fig, use_container_width=True)
