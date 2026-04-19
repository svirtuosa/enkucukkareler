import streamlit as st
import numpy as np
import plotly.graph_objects as go
import re

# --- SAYFA VE HAFIZA AYARLARI ---
st.set_page_config(page_title="EKK vs QR Simülasyonu", layout="wide")

if "fonk_metni" not in st.session_state:
    st.session_state.fonk_metni = "np.sin(x) + np.cos(2*x)"

def ekle_metin(ek):
    st.session_state.fonk_metni += ek

# --- BAŞLIK ---
st.title("📈 Polinom Uydurma: EKK vs QR")
st.markdown("Bu laboratuvarda Klasik EKK'nın yüksek derecelerdeki çöküşünü ve QR ayrışımının kararlılığını test edeceğiz.")
st.write("---")

# ==========================================
# ADIM 1: KULLANICIYA YÖNTEM SEÇTİRME
# ==========================================
st.subheader("Adım 1: Veri Giriş Yöntemi")

mod = st.radio(
    "Lütfen verilerinizi nasıl oluşturmak istediğinizi seçin:", 
    ["🔢 Noktaları Manuel Belirle (Barlar ile)", "📐 Matematiksel Fonksiyon Yaz"],
    index=None,
    horizontal=True
)

if mod is None:
    st.info("👆 Lütfen simülasyona başlamak için yukarıdan bir yöntem seçin.")
    st.stop()

st.write("---")

x_verisi = np.array([])
y_verisi = np.array([])

# ==========================================
# ADIM 2: SEÇİLEN MODA GÖRE EKRAN GÖSTERİMİ
# ==========================================
if mod == "🔢 Noktaları Manuel Belirle (Barlar ile)":
    st.subheader("Adım 2: Noktaları Ayarla")
    
    nokta_sayisi = st.slider("Kaç Adet Nokta (Değişken) Olsun?", min_value=3, max_value=15, value=6)
    st.markdown("**Noktaların Y Değerlerini Aşağıdaki Kaydırıcılardan Belirleyin:**")
    
    kolonlar = st.columns(nokta_sayisi)
    y_verisi_list = []
    x_verisi = np.arange(nokta_sayisi, dtype=float)
    
    for i in range(nokta_sayisi):
        with kolonlar[i]:
            varsayilan = float(np.sin(i) * 5 + 5)
            val = st.slider(f"X={i}", min_value=-10.0, max_value=20.0, value=varsayilan, step=0.5, label_visibility="collapsed")
            y_verisi_list.append(val)
            st.markdown(f"<div style='text-align: center; color: gray; font-size:14px; font-weight:bold;'>X={i}</div>", unsafe_allow_html=True)
            
    y_verisi = np.array(y_verisi_list)

elif mod == "📐 Matematiksel Fonksiyon Yaz":
    st.subheader("Adım 2: Fonksiyonu Tanımla")
    
    with st.container(border=True):
        st.text_input("f(x) =", key="fonk_metni")
        
        st.caption("Hızlı Ekleme Butonları:")
        b1, b2, b3, b4, b5, b6, b7, b8, _ = st.columns([1, 1, 1, 1, 1, 1.5, 1.5, 1.5, 2])
        
        b1.button("x²", on_click=ekle_metin, args=("x^2",))
        b2.button("x³", on_click=ekle_metin, args=("x^3",))
        b3.button("x⁴", on_click=ekle_metin, args=("x^4",))
        b4.button("x⁵", on_click=ekle_metin, args=("x^5",))
        b5.button(" + ", on_click=ekle_metin, args=(" + ",))
        b6.button("sin(x)", on_click=ekle_metin, args=("np.sin(x)",))
        b7.button("cos(x)", on_click=ekle_metin, args=("np.cos(x)",))
        b8.button("exp(x)", on_click=ekle_metin, args=("np.exp(x)",))
        
        st.write("") 
        f_nokta = st.number_input("Grafikte Çizilecek Nokta Sayısı (Çözünürlük)", min_value=10, max_value=200, value=50)
            
    x_verisi = np.linspace(-5, 5, f_nokta)
    
    guvenli_fonksiyon = st.session_state.fonk_metni.replace("^", "**")
    guvenli_fonksiyon = re.sub(r'(\d)\s*(x|np)', r'\1*\2', guvenli_fonksiyon)
    
    try:
        y_verisi = eval(guvenli_fonksiyon, {"np": np, "x": x_verisi})
        if isinstance(y_verisi, (int, float)):
            y_verisi = np.full_like(x_verisi, y_verisi)
    except Exception:
        st.error("⚠️ Geçersiz fonksiyon! Lütfen denkleminizi kontrol edin.")
        y_verisi = np.zeros(f_nokta)

st.write("---")

# ==========================================
# ADIM 3: POLİNOM DERECESİ VE GRAFİK EKRANI
# ==========================================
st.subheader("Adım 3: Polinom Uydurma ve Karşılaştırma")

derece = st.slider("Uydurulacak Polinom Derecesi (EKK'nın Çöküşünü Görmek İçin Artırın):", min_value=1, max_value=20, value=3)

if len(x_verisi) <= derece:
    st.warning(f"⚠️ {derece}. dereceden polinom için en az {derece + 1} nokta gereklidir. Lütfen yukarıdan nokta sayısını artırın veya dereceyi düşürün.")
    st.stop()

# --- MATEMATİKSEL HESAPLAMALAR ---
A = np.vander(x_verisi, N=derece + 1, increasing=True)

# Klasik EKK
try:
    beta_ekk = np.linalg.inv(A.T @ A) @ A.T @ y_verisi
    y_tahmin_ekk = A @ beta_ekk
    mse_ekk = np.mean((y_verisi - y_tahmin_ekk)**2) # Hata Skoru Hesaplama
    ekk_basarili = True
except np.linalg.LinAlgError:
    ekk_basarili = False
    mse_ekk = float('inf')

# QR Ayrışımı
Q, R = np.linalg.qr(A)
beta_qr = np.linalg.solve(R, Q.T @ y_verisi)
y_tahmin_qr = A @ beta_qr
mse_qr = np.mean((y_verisi - y_tahmin_qr)**2) # Hata Skoru Hesaplama

# --- HATA SKORU GÖSTERİMİ (MİNİK KUTULAR) ---
# Bilimsel gösterim (.2e) kullanıyoruz çünkü EKK çöktüğünde sayılar milyarlara ulaşacak
st.markdown("**Hata Skoru Karşılaştırması (MSE)**")
kutu1, kutu2, bosluk = st.columns([1, 1, 2])

with kutu1:
    if ekk_basarili:
        st.metric(label="Klasik EKK Hatası", value=f"{mse_ekk:.2e}")
    else:
        st.metric(label="Klasik EKK Hatası", value="SİSTEM ÇÖKTÜ")

with kutu2:
    st.metric(label="QR Ayrışımı Hatası", value=f"{mse_qr:.2e}")

# --- GÖRSELLEŞTİRME ---
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x_verisi, y=y_verisi, mode='markers', 
    marker=dict(symbol='diamond', size=8, color='white', line=dict(width=1.5, color='black')), 
    name='Veri Noktaları'
))

x_min, x_max = x_verisi.min(), x_verisi.max()
x_cizgi = np.linspace(x_min - 0.5, x_max + 0.5, 300)
A_cizgi = np.vander(x_cizgi, N=derece + 1, increasing=True)

if ekk_basarili:
    fig.add_trace(go.Scatter(x=x_cizgi, y=A_cizgi @ beta_ekk, mode='lines', line=dict(color='blue', width=4), name='Klasik EKK'))

fig.add_trace(go.Scatter(x=x_cizgi, y=A_cizgi @ beta_qr, mode='lines', line=dict(color='indianred', width=4, dash='dash'), name='QR Ayrışımı'))

y_min, y_max = y_verisi.min(), y_verisi.max()
fark = max(y_max - y_min, 1) 

fig.update_layout(
    xaxis_title="X Ekseni", 
    yaxis_title="Y Ekseni",
    yaxis_range=[y_min - fark*0.3, y_max + fark*0.3], 
    margin=dict(t=30, b=10),
    template="plotly_white",
    height=500,
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
)

st.plotly_chart(fig, use_container_width=True)
