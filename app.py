import streamlit as st
import numpy as np
import plotly.graph_objects as go
import re

st.set_page_config(page_title="EKK vs QR Simülasyonu", layout="wide")

# --- SESSION STATE (Butonların metin kutusuna yazabilmesi için gerekli hafıza) ---
if "fonk_metni" not in st.session_state:
    st.session_state.fonk_metni = "x^2 + np.sin(x)"

def ekle_metin(ek):
    """Butona basıldığında metin kutusunun sonuna seçilen ifadeyi ekler."""
    st.session_state.fonk_metni += ek

# --- BAŞLIK ---
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
    nokta_sayisi = st.slider("Kaç Adet Nokta (Değişken) Olsun?", min_value=3, max_value=15, value=6)
    st.markdown("**Noktaların Y Değerlerini Ayarlayın:**")
    
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

# --- MOD 2: MATEMATİKSEL FONKSİYON VE BUTONLAR ---
else:
    with st.container(border=True):
        st.markdown("**Fonksiyon Ayarları**")
        
        # Metin kutusu (session_state'e bağlı)
        st.text_input("f(x) =", key="fonk_metni")
        
        # HIZLI EKLEME BUTONLARI
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
        
        st.write("") # Boşluk
        f_nokta = st.number_input("Grafikte Çizilecek Nokta Sayısı (Çözünürlük)", min_value=10, max_value=200, value=50)
            
    x_verisi = np.linspace(-5, 5, f_nokta)
    
    # --- ARKA PLAN DÜZELTMELERİ (Kullanıcı Dostu Python) ---
    # 1. Kullanıcının ^ işaretini Python'un anladığı ** işaretine çevir
    guvenli_fonksiyon = st.session_state.fonk_metni.replace("^", "**")
    
    # 2. "4x" yazılmışsa bunu "4*x" veya "4np.sin" yazılmışsa "4*np.sin" yap (Regex ile gizli çarpım eklentisi)
    guvenli_fonksiyon = re.sub(r'(\d)\s*(x|np)', r'\1*\2', guvenli_fonksiyon)
    
    try:
        y_verisi = eval(guvenli_fonksiyon, {"np": np, "x": x_verisi})
        # Eğer y_verisi sabit bir sayı çıkarsa (örneğin sadece "5" yazıldıysa) array'e çevir
        if isinstance(y_verisi, (int, float)):
            y_verisi = np.full_like(x_verisi, y_verisi)
    except Exception:
        st.error("⚠️ Geçersiz fonksiyon! Lütfen denkleminizi kontrol edin.")
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

# Ham Veri Noktaları
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
