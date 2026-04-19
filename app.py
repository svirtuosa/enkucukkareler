import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="EKK vs QR Simülasyonu", layout="wide")

st.title("🚨 Klasik EKK'nın Çöküşü ve QR'ın Gücü")
st.markdown("""
Aşağıdaki sekmelerden veriyi nasıl oluşturmak istediğinizi seçin. Ardından sol üstten polinom derecesini 
artırarak Klasik EKK'nın bilgisayar yuvarlama hataları yüzünden nasıl çöktüğünü gözlemleyin.
""")
st.write("---")

# --- ÜST AYAR BARI ---
sol_ayar, sag_ayar = st.columns([1, 2])
with sol_ayar:
    derece = st.slider("Uydurulacak Polinom Derecesi:", min_value=1, max_value=20, value=3, step=1)
    veri_kaynagi = st.radio("Kullanılacak Veri Seti:", ["Manuel Tablo", "Fonksiyon"], horizontal=True)

# --- SEKMELER (TABS) ---
sekme1, sekme2 = st.tabs(["✍️ Tablo ile Manuel Giriş", "📐 Fonksiyondan Veri Üret"])

x_verisi = np.array([])
y_verisi = np.array([])

with sekme1:
    st.subheader("Kendi Verilerini Gir")
    
    # Tablodaki boşluğu kırpmak için ekranı bölüyoruz (Tablo 1 birim, boşluk 2 birim)
    col_tablo, col_bosluk = st.columns([1, 2])
    
    with col_tablo:
        varsayilan_veri = pd.DataFrame({
            "X": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            "Y": [0.2, 0.8, 1.1, 0.9, -0.2, -0.9, -1.2, -0.5, 0.4, 0.9, 1.3]
        })
        
        # hide_index=True ile daha temiz bir görünüm sağlıyoruz
        duzenlenmis_veri = st.data_editor(varsayilan_veri, num_rows="dynamic", use_container_width=True, hide_index=True)
        x_manuel = duzenlenmis_veri["X"].to_numpy(dtype=float)
        y_manuel = duzenlenmis_veri["Y"].to_numpy(dtype=float)
        
    with col_bosluk:
        st.info("👈 Tabloya çift tıklayarak verileri değiştirebilir, en alta inerek yeni satır ekleyebilirsiniz.")

with sekme2:
    st.subheader("Matematiksel Fonksiyon ile Üret")
    st.info("Geçerli bir matematiksel ifade yazın (örnek: `np.sin(x) + x^2`). `^` işareti arka planda otomatik düzeltilecektir.")
    
    fonksiyon_metni = st.text_input("f(x) = ", value="np.sin(x) + x^2")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        nokta_sayisi = st.number_input("Nokta Sayısı", min_value=5, max_value=500, value=50)
    with col_b:
        x_araligi = st.slider("X Ekseni Aralığı", min_value=-10.0, max_value=10.0, value=(0.0, 5.0))
    with col_c:
        gurultu = st.slider("Gürültü (Noise) Miktarı", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
        
    x_fonksiyon = np.linspace(x_araligi[0], x_araligi[1], nokta_sayisi)
    guvenli_fonksiyon = fonksiyon_metni.replace("^", "**")
    
    try:
        y_fonksiyon = eval(guvenli_fonksiyon, {"np": np, "x": x_fonksiyon})
        np.random.seed(42)
        y_fonksiyon += np.random.normal(0, gurultu, nokta_sayisi)
    except Exception as e:
        st.error("⚠️ Fonksiyon anlaşılamadı. Lütfen 'np.sin(x)' veya 'x^2' gibi geçerli bir format kullanın.")
        y_fonksiyon = np.zeros(nokta_sayisi)

# --- VERİ SEÇİMİ ---
if veri_kaynagi == "Manuel Tablo":
    x_verisi, y_verisi = x_manuel, y_manuel
else:
    x_verisi, y_verisi = x_fonksiyon, y_fonksiyon

# --- MATEMATİKSEL HESAPLAMALAR ---
if len(x_verisi) <= derece:
    st.warning(f"⚠️ {derece}. dereceden polinom için en az {derece + 1} nokta gereklidir. Lütfen tabloya veri ekleyin veya dereceyi düşürün.")
    st.stop()

A = np.vander(x_verisi, N=derece + 1, increasing=True)

try:
    beta_ekk = np.linalg.inv(A.T @ A) @ A.T @ y_verisi
    ekk_basarili = True
except np.linalg.LinAlgError:
    ekk_basarili = False
    st.error("⚠️ Klasik EKK Çöktü: Matris tersinir değil (Singular Matrix)!")

Q, R = np.linalg.qr(A)
beta_qr = np.linalg.solve(R, Q.T @ y_verisi)

# --- GÖRSELLEŞTİRME ---
fig = go.Figure()

fig.add_trace(go.Scatter(x=x_verisi, y=y_verisi, mode='markers', 
                         marker=dict(size=8, color='black', opacity=0.7), name='Ham Veri Noktaları'))

x_min, x_max = x_verisi.min(), x_verisi.max()
x_cizgi = np.linspace(x_min - 0.5, x_max + 0.5, 300)
A_cizgi = np.vander(x_cizgi, N=derece + 1, increasing=True)

if ekk_basarili:
    y_cizgi_ekk = A_cizgi @ beta_ekk
    fig.add_trace(go.Scatter(x=x_cizgi, y=y_cizgi_ekk, mode='lines', 
                             line=dict(color='red', width=4), name='Klasik EKK (Çökmeye Meyilli)'))

y_cizgi_qr = A_cizgi @ beta_qr
fig.add_trace(go.Scatter(x=x_cizgi, y=y_cizgi_qr, mode='lines', 
                         line=dict(color='blue', width=4, dash='dot'), name='QR Ayrışımı (Kararlı)'))

y_min, y_max = y_verisi.min(), y_verisi.max()
fark = max(y_max - y_min, 1) 
fig.update_layout(
    title=f"Derece: {derece} | EKK ve QR Çözümlerinin Karşılaştırması", 
    xaxis_title="X Ekseni", 
    yaxis_title="Y",
    yaxis_range=[y_min - fark*0.5, y_max + fark*0.5], 
    margin=dict(t=50),
    template="plotly_white",
    height=600
)

st.plotly_chart(fig, use_container_width=True)
