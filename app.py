import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Sayfa ayarları
st.set_page_config(page_title="EKK Çöküşü vs QR", layout="wide")

# Başlık ve Açıklama
st.title("🚨 Klasik EKK'nın Çöküşü ve QR'ın Gücü")
st.markdown("""
Polinom derecesi arttıkça, Klasik EKK yönteminde kullanılan $A^T A$ matrisinin 
**kondisyon sayısı** karesi oranında büyür. Bu durum bilgisayarın yuvarlama hatalarını patlatır. 
Aşağıdaki kaydırıcıdan dereceyi artırarak EKK'nın nasıl çöktüğünü izleyin.
""")

st.write("---")

# --- ÜST AYAR BARI (Sol Üstte Minik Bar) ---
# Ekranı 1'e 3 oranında bölerek slider'ı sadece sol tarafa sıkıştırıyoruz
sol_ayar, bos_kisim = st.columns([1, 3])

with sol_ayar:
    derece = st.slider("Polinom Derecesi:", min_value=1, max_value=15, value=2, step=1)

# --- MATEMATİKSEL ALTYAPI ---
nokta_sayisi = 30

# Rastgele veri üretimi (sinüs dalgası + gürültü)
np.random.seed(42) 
x_verisi = np.linspace(0, 5, nokta_sayisi)
y_verisi = np.sin(x_verisi) + np.random.normal(0, 0.2, nokta_sayisi)

# Vandermonde Matrisi (A matrisi)
A = np.vander(x_verisi, N=derece + 1, increasing=True)

# 1. Klasik EKK Çözümü
try:
    beta_ekk = np.linalg.inv(A.T @ A) @ A.T @ y_verisi
    ekk_basarili = True
except np.linalg.LinAlgError:
    ekk_basarili = False
    st.error("⚠️ Klasik EKK Çöktü: Matris tersinir değil (Singular Matrix)!")

# 2. QR Ayrışımı Çözümü
Q, R = np.linalg.qr(A)
beta_qr = np.linalg.solve(R, Q.T @ y_verisi)

# --- GÖRSELLEŞTİRME ---
fig = go.Figure()

# Gerçek Noktalar
fig.add_trace(go.Scatter(x=x_verisi, y=y_verisi, mode='markers', 
                         marker=dict(size=8, color='black'), name='Gerçek Veriler'))

x_cizgi = np.linspace(0, 5, 200)
A_cizgi = np.vander(x_cizgi, N=derece + 1, increasing=True)

# EKK Doğrusu
if ekk_basarili:
    y_cizgi_ekk = A_cizgi @ beta_ekk
    fig.add_trace(go.Scatter(x=x_cizgi, y=y_cizgi_ekk, mode='lines', 
                             line=dict(color='red', width=4), name='Klasik EKK (Çökmeye Meyilli)'))

# QR Doğrusu
y_cizgi_qr = A_cizgi @ beta_qr
fig.add_trace(go.Scatter(x=x_cizgi, y=y_cizgi_qr, mode='lines', 
                         line=dict(color='blue', width=4, dash='dot'), name='QR Ayrışımı (Kararlı)'))

# Grafik Ayarları
fig.update_layout(
    title=f"Derece: {derece} | EKK vs QR", 
    xaxis_title="X", 
    yaxis_title="Y",
    yaxis_range=[-2, 3], # Sapmaları net görmek için ekseni kilitledik
    margin=dict(t=40) 
)

# Grafiği ekrana bas
st.plotly_chart(fig, use_container_width=True)
