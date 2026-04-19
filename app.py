import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Polinom Uydurma", layout="wide")

st.title("Polinom Uydurma: EKK vs QR")
st.markdown("Noktaların Y değerlerini aşağıdaki çubuklardan değiştirerek doğrunun anlık tepkisini izleyin.")

# Üst Ayarlar
col_ayar, _ = st.columns([1, 2])
with col_ayar:
    derece = st.slider("Polinom Derecesi", min_value=1, max_value=10, value=3)

st.write("---")

# Nokta Sayısı ve Başlangıç Değerleri
nokta_sayisi = 8
x_verisi = np.arange(nokta_sayisi, dtype=float)

# Dikey Slider'lar için kolonlar (Ekolayzer görünümü)
kolonlar = st.columns(nokta_sayisi)
y_verisi = np.zeros(nokta_sayisi)

baslangic_y = [2.0, 4.5, 3.1, 8.0, 6.2, 9.5, 4.0, 1.0]

st.markdown("**Noktaları Hareket Ettirin (Y Değerleri):**")
for i in range(nokta_sayisi):
    with kolonlar[i]:
        # label_visibility="collapsed" ile başlıkları gizleyip çok temiz bir görüntü alıyoruz
        y_verisi[i] = st.slider(
            f"X={i}", 
            min_value=-5.0, 
            max_value=15.0, 
            value=baslangic_y[i], 
            step=0.5,
            label_visibility="collapsed" 
        )

# --- MATEMATİKSEL HESAPLAMALAR ---
A = np.vander(x_verisi, N=derece + 1, increasing=True)

try:
    beta_ekk = np.linalg.inv(A.T @ A) @ A.T @ y_verisi
    ekk_basarili = True
except np.linalg.LinAlgError:
    ekk_basarili = False

Q, R = np.linalg.qr(A)
beta_qr = np.linalg.solve(R, Q.T @ y_verisi)

# --- GÖRSELLEŞTİRME ---
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x_verisi, y=y_verisi, mode='markers', 
    marker=dict(symbol='diamond', size=12, color='white', line=dict(width=2, color='black')), 
    name='Veri Noktaları'
))

x_cizgi = np.linspace(-0.5, nokta_sayisi - 0.5, 200)
A_cizgi = np.vander(x_cizgi, N=derece + 1, increasing=True)

if ekk_basarili:
    fig.add_trace(go.Scatter(x=x_cizgi, y=A_cizgi @ beta_ekk, mode='lines', 
                             line=dict(color='blue', width=4, shape='spline'), name='Klasik EKK'))

fig.add_trace(go.Scatter(x=x_cizgi, y=A_cizgi @ beta_qr, mode='lines', 
                         line=dict(color='indianred', width=4, dash='dash', shape='spline'), name='QR Ayrışımı'))

fig.update_layout(
    xaxis_title="X Ekseni", 
    yaxis_title="Y Ekseni",
    yaxis_range=[-6, 16], # Eksenleri sabit tutmak kayma hissini artırır
    margin=dict(t=20, b=20),
    template="plotly_white",
    height=450
)

st.plotly_chart(fig, use_container_width=True)
