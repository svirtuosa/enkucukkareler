import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="EKK vs QR Simülasyonu", layout="wide")

st.title("📈 Polinom Uydurma: EKK vs QR")
st.markdown("Aşağıdan veri giriş modunu seçin ve polinom derecesini ayarlayarak EKK'nın yüksek derecelerdeki çöküşünü izleyin.")

# --- ÜST AYARLAR ---
col_derece, col_mod, bosluk = st.columns([1, 1, 2])
with col_derece:
    derece = st.slider("Polinom Derecesi:", min_value=1, max_value=15, value=3)
with col_mod:
    mod = st.radio("Veri Giriş Yöntemi:", ["Manuel Noktalar (Barlar)", "Matematiksel Fonksiyon"])

st.write("---")

x_verisi = np.array([])
y_verisi = np.array([])

# --- MOD 1: MANUEL NOKTALAR ---
if mod == "Manuel Noktalar (Barlar)":
    # Kullanıcı kaç nokta istiyorsa onu seçiyor
    nokta_sayisi = st.slider("Kaç Adet Nokta (Değişken) Olsun?", min_value=3, max_value=15, value=6)
    
    st.markdown("**Noktaların Y Değerlerini Ayarlayın:**")
    
    # Seçilen nokta sayısı kadar yan yana dinamik kolon oluştur
    kolonlar = st.columns(nokta_sayisi)
    y_verisi_list = []
    x_verisi = np.arange(nokta_sayisi, dtype=float)
    
    for i in range(nokta_sayisi):
        with kolonlar[i]:
            # Varsayılan başlangıç değeri (hoş bir eğri oluşturması için sinüs kullandık)
            varsayilan = float(np.sin(i) * 5 + 5)
            
            # label_visibility="collapsed" ile kaydırıcıların başlıklarını gizleyip temiz yapıyoruz
            val = st.slider(
                f"X={i}", 
                min_value=-10.0, 
                max_value=20.0, 
                value=varsayilan, 
                step=0.5, 
                label_visibility="collapsed"
            )
            y_verisi_list.append(val)
            # Altlarına minik X etiketleri yazıyoruz
            st.markdown(f"<div style='text-align: center; color: gray; font-size:14px; font-weight:bold;'>X={i}</div>", unsafe_allow_html=True)
            
    y_verisi = np.array(y_verisi_list)

# --- MOD 2: MATEMATİKSEL FONKSİYON ---
else:
    with st.container(border=True):
        st.markdown("**Fonksiyon Ayarları**")
        f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
        
        with f_col1:
            fonksiyon_metni = st.text_input("f(x) =", value="np.sin(x) * 5 + x^2")
            st.caption("Örn: np.sin(x), np.cos(x), x^2")
        with f_col2:
            f_nokta = st.number_input("Oluşturulacak Nokta Sayısı", min_value=10, max_value=200, value=50)
        with f_col3:
            gurultu = st.slider("Gürültü (Noise)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
            
    x_verisi = np.linspace(-5, 5, f_nokta)
    
    # ^ işaretini Python'un anlayacağı ** işaretine çeviriyoruz
    guvenli_fonksiyon = fonksiyon_metni.replace("^", "**")
    
    try:
        y_fonksiyon = eval(guvenli_fonksiyon, {"np": np, "x": x_verisi})
        np.random.seed(42)
        y_fonksiyon += np.random.normal(0, gurultu, f_nokta)
        y_verisi = y_fonksiyon
    except Exception:
        st.error("⚠️ Geçersiz fonksiyon! Lütfen np.sin(x) veya x^2 gibi bir format kullanın.")
        y_verisi = np.zeros(f_nokta)

# --- HESAPLAMA VE GÜVENLİK KONTROLÜ ---
if len(x_verisi) <= derece:
    st.warning(f"⚠️ {derece}. dereceden polinom için en az {derece + 1} nokta gereklidir. Lütfen nokta sayısını artırın veya dereceyi düşürün.")
    st.stop()

# Tasarım Matrisi
A = np.vander(x_verisi, N=derece + 1, increasing=True)

# Klasik EKK
try:
    beta_ekk = np.linalg.inv(A.T @ A) @ A.T @ y_verisi
    ekk_basarili = True
except np.linalg.LinAlgError:
    ekk_basarili = False

# QR Ayrışımı
Q, R = np.linalg.qr(A)
beta_qr = np.linalg.solve(R, Q.T @ y_verisi)

# --- GÖRSELLEŞTİRME ---
fig = go.Figure()

# Noktalar (Elmas şeklinde)
fig.add_trace(go.Scatter(
    x=x_verisi, y=y_verisi, mode='markers', 
    marker=dict(symbol='diamond', size=8, color='white', line=dict(width=1.5, color='black')), 
    name='Veri Noktaları'
))

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
    xaxis_title="X Ekseni", 
    yaxis_title="Y Ekseni",
    yaxis_range=[y_min - fark*0.3, y_max + fark*0.3], 
    margin=dict(t=30, b=10),
    template="plotly_white",
    height=500,
    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
)

st.plotly_chart(fig, use_container_width=True)
