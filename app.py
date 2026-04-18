import streamlit as st
import numpy as np
import plotly.graph_objects as go

# 1. Sayfa Ayarları
st.set_page_config(page_title="QR Ayrışımı Projesi", layout="wide")

# Başlıklar
st.title("🔢 En Küçük Kareler ve QR Ayrışımı")
st.subheader("Numerik Analiz Projesi")

# 2. Yan Menü (Sidebar) üzerinden veri girişi
st.sidebar.header("Veri Girişi")
st.sidebar.write("Y noktalarını kaydırarak değiştirin:")

x_verisi = np.array([0, 1, 2, 3, 4])
y_verisi = []

for i in range(5):
    deger = st.sidebar.slider(f"Nokta {i+1} (x={i})", 0.0, 10.0, float(i + 2), 0.1)
    y_verisi.append(deger)

y_verisi = np.array(y_verisi)

# 3. QR Ayrışımı ile Hesaplama
# Tasarım matrisi A = [[1, x0], [1, x1], ...]
A = np.vstack([np.ones(len(x_verisi)), x_verisi]).T

# A = QR ayrışımı
Q, R = np.linalg.qr(A)

# R * beta = Q.T * y denklemini çöz (En küçük kareler çözümü)
beta = np.linalg.solve(R, Q.T @ y_verisi)
sabit_terim, egim = beta

# Tahmin değerleri
y_tahmin = A @ beta

# 4. Görselleştirme (Plotly)
fig = go.Figure()

# Gerçek Noktalar
fig.add_trace(go.Scatter(x=x_verisi, y=y_verisi, mode='markers', 
                         marker=dict(size=12, color='red'), name='Veri Noktaları'))

# Regresyon Doğrusu
x_cizgi = np.linspace(-0.5, 4.5, 100)
y_cizgi = sabit_terim + egim * x_cizgi
fig.add_trace(go.Scatter(x=x_cizgi, y=y_cizgi, mode='lines', 
                         line=dict(color='blue', width=3), name='Uydurulan Doğru'))

# Hata (Residual) Çizgileri
for i in range(len(x_verisi)):
    fig.add_trace(go.Scatter(x=[x_verisi[i], x_verisi[i]], y=[y_verisi[i], y_tahmin[i]],
                             mode='lines', line=dict(color='gray', dash='dash'), showlegend=False))

fig.update_layout(title="QR Ayrışımı ile En Uygun Doğru", xaxis_title="X", yaxis_title="Y")

# 5. Ekrana Yazdırma
sol_kolon, sag_kolon = st.columns([2, 1])

with sol_kolon:
    st.plotly_chart(fig, use_container_width=True)

with sag_kolon:
    st.write("### Matematiksel Model")
    st.latex(rf"y = {sabit_terim:.3f} + {egim:.3f}x")
    
    st.write("---")
    st.write("**QR Ayrışımı Detayları:**")
    st.write("Q Matrisi (Ortogonal):")
    st.dataframe(Q)
    st.write("R Matrisi (Üst Üçgen):")
    st.dataframe(R)

st.success("Proje başarıyla çalışıyor! Sol taraftaki menüden noktaları değiştirebilirsiniz.")
