import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Polinom Uydurma: EKK vs QR", layout="wide")

# --- BAŞLIK ---
st.title("Polinom Uydurma: EKK vs QR")
st.markdown("Polinom uydurma deneyinde yüksek derecelerde Klasik EKK yönteminin çöküşünü ($A^T A$ matrisindeki yuvarlama hataları nedeniyle), QR ayrışımının ise kararlılığını simüle edin.")

st.write("") # Boşluk

# --- ANA DÜZEN (1/3 ve 2/3 Oranında İki Kolon) ---
col1, col2 = st.columns([1, 2])

# SOL KOLON: Veri Kaynağı Seçimi ve Tablo
with col1:
    veri_kaynagi = st.radio("Veri Kaynağı:", ["Manuel Tablo", "Fonksiyon"], horizontal=True)
    
    varsayilan_veri = pd.DataFrame({
        "X": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "Y": [10.9, 16.7, 25.3, 36.0, 20.1, 15.5]
    })
    
    # Kullanıcı 'Fonksiyon' seçerse tablo pasif (gri) olur
    duzenlenmis_veri = st.data_editor(
        varsayilan_veri, 
        num_rows="dynamic", 
        use_container_width=True, 
        hide_index=True,
        disabled=(veri_kaynagi == "Fonksiyon")
    )
    x_manuel = duzenlenmis_veri["X"].to_numpy(dtype=float)
    y_manuel = duzenlenmis_veri["Y"].to_numpy(dtype=float)

# SAĞ KOLON: Polinom Derecesi ve Fonksiyon Ayarları
with col2:
    derece = st.slider("Polinom Derecesi:", min_value=1, max_value=15, value=13, step=1)
    
    # Etrafı çerçeveli şık bir kutu
    with st.container(border=True):
        st.markdown("**Fonksiyon**")
        
        # Kullanıcı 'Manuel Tablo' seçerse buralar pasif (gri) olur
        is_disabled = (veri_kaynagi == "Manuel Tablo")
        
        fonksiyon_metni = st.text_input("f(x) =", value="np.sin(x) + x^2", disabled=is_disabled)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            nokta_sayisi = st.number_input("Nokta Sayısı", min_value=5, max_value=500, value=50, disabled=is_disabled)
        with c2:
            x_araligi = st.slider("X Ekseni Aralığı", -10.0, 10.0, (0.0, 5.0), disabled=is_disabled)
        with c3:
            gurultu = st.slider("Gürültü (Noise) Miktarı", 0.0, 5.0, 0.5, step=0.1, disabled=is_disabled)

        x_fonksiyon = np.linspace(x_araligi[0], x_araligi[1], nokta_sayisi)
        guvenli_fonksiyon = fonksiyon_metni.replace("^", "**")
        
        try:
            y_fonksiyon = eval(guvenli_fonksiyon, {"np": np, "x": x_fonksiyon})
            np.random.seed(42)
            y_fonksiyon += np.random.normal(0, gurultu, nokta_sayisi)
        except Exception:
            y_fonksiyon = np.zeros(nokta_sayisi)

# --- VERİ BİRLEŞTİRME ---
if veri_kaynagi == "Manuel Tablo":
    x_verisi, y_verisi = x_manuel, y_manuel
else:
    x_verisi, y_verisi = x_fonksiyon, y_fonksiyon

if len(x_verisi) <= derece:
    st.warning(f"⚠️ {derece}. dereceden polinom için en az {derece + 1} nokta gereklidir.")
    st.stop()

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

# Görseldeki gibi elmas (diamond) şekilli içi beyaz noktalar
fig.add_trace(go.Scatter(x=x_verisi, y=y_verisi, mode='markers', 
                         marker=dict(symbol='diamond', size=8, color='white', line=dict(width=1.5, color='black')), 
                         name='Ham Veri Noktaları'))

x_min, x_max = x_verisi.min(), x_verisi.max()
x_cizgi = np.linspace(x_min - 0.5, x_max + 0.5, 300)
A_cizgi = np.vander(x_cizgi, N=derece + 1, increasing=True)

if ekk_basarili:
    fig.add_trace(go.Scatter(x=x_cizgi, y=A_cizgi @ beta_ekk, mode='lines', 
                             line=dict(color='blue', width=4), name='Klasik EKK'))

fig.add_trace(go.Scatter(x=x_cizgi, y=A_cizgi @ beta_qr, mode='lines', 
                         line=dict(color='indianred', width=4, dash='dash'), name='QR Ayrışımı'))

y_min, y_max = y_verisi.min(), y_verisi.max()
fark = max(y_max - y_min, 1) 

fig.update_layout(
    title=dict(text=f"Derece {derece} | EKK ve QR Çözümlerinin Karşılaştırması", x=0.5), 
    xaxis_title="X Ekseni", 
    yaxis_title="Y",
    yaxis_range=[y_min - fark*0.5, y_max + fark*0.5], 
    margin=dict(t=50, b=10),
    template="plotly_white",
    height=450,
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
)

st.plotly_chart(fig, use_container_width=True)

# --- ALT BİLGİ KUTUSU ---
with st.expander("➕ Teknik Bilgi"):
    st.markdown("A matrisini ortogonal Q ve üst üçgen R matrislerine ayıran QR yöntemi, $A^T A$ çarpımından kaçındığı için bilgisayarın yuvarlama hatalarına karşı çok daha dirençlidir. Klasik EKK, yüksek derecelerde kararsız ve tekil ($singular$) matrisler üretir.")
