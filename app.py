import streamlit as st
import numpy as np
import plotly.graph_objects as go
import re
import base64  # Yeni eklenen: Arka plan görselini CSS'ye gömmek için gerekli

# --- ARKA PLAN GÖRSELİNİ CSS FİLTRESİ İLE AYARLAMA İŞLEVİ ---
def set_dark_background_css(image_path, brightness=0.25): # brightness 0.0-1.0 arası, 0.25 = %75 koyu
    try:
        # Görseli base64'e çevir
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        # CSS kodu oluştur (brightness filtresi ile)
        #data-testid="stAppViewContainer" ve .main > div > .block-container kısımları Streamlit'in yapısıdır.
        page_bg_css = f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            filter: brightness({brightness});
        }}
        
        [data-testid="stAppViewContainer"] > .main > div {{
            position: relative; /* İçeriğin filtreden etkilenmemesi için */
            z-index: 1;
        }}
        
        /* Grafiklerin ve metinlerin koyu arka plan üzerinde daha net görünmesi için renk ayarları */
        .stMetric, .stPlotlyChart, .stDataFrame, .stAlert, .stText, h1, h2, h3, h4, h5, h6, p, label {{
            color: #f0f2f6 !important; /* Açık gri/beyaz metin */
        }}
        
        .stSlider, .stRadio, .stTextInput, .stNumberInput, .stButton, .st-bo, .st-c2 {{
            background-color: rgba(30, 34, 42, 0.7); /* Koyu, yarı şeffaf kutular */
            color: #f0f2f6 !important;
            border-radius: 8px;
            padding: 5px;
        }}
        
        .stExpander, .stTabs {{
            background-color: rgba(30, 34, 42, 0.5); /* Daha koyu, yarı şeffaf sekmeler */
            border-radius: 8px;
        }}
        
        .stMarkdown div p {{
            color: #f0f2f6 !important;
        }}
        
        /* Plotly grafiklerinin arka planını şeffaf yap */
        .js-plotly-plot .plotly .bg {{
            fill: transparent !important;
        }}
        </style>
        """
        st.markdown(page_bg_css, unsafe_allow_html=True)
        
    except FileNotFoundError:
        st.error(f"⚠️ Arka plan görseli bulunamadı: {image_path}. Lütfen dosya yolunu kontrol edin.")

# --- SAYFA VE HAFIZA AYARLARI ---
st.set_page_config(page_title="EKK vs QR Simülasyonu", layout="wide")

# Fonksiyon girişi için buton hafızasını hazırlıyoruz
if "fonk_metni" not in st.session_state:
    st.session_state.fonk_metni = "np.sin(x) + np.cos(2*x)"

def ekle_metin(ek):
    st.session_state.fonk_metni += ek

# --- ARKA PLANI AYARLA (Kodun başına ekle) ---
# Görsel dosyasının adı image_13.png olarak varsayıyoruz, projenin kök klasöründe olmalı.
# brightness=0.25 varsayıyoruz, bu %75 koyulaştırma demektir.
set_dark_background_css("image_13.png", brightness=0.2) # Daha da koyu olsun (%80)

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
# Renkleri açık gri yapmak için st.markdown ile manuel stil uyguluyoruz
k1.markdown(f'<p style="color:#f0f2f6;">Klasik EKK Hatası (MSE)</p><p style="color:#f0f2f6; font-size:30px; font-weight:bold;">{mse_ekk:.2e}</p>' if ekk_basarili else '<p style="color:#f0f2f6;">Klasik EKK Hatası (MSE)</p><p style="color:red; font-size:30px; font-weight:bold;">ÇÖKTÜ</p>', unsafe_allow_html=True)
k2.markdown(f'<p style="color:#f0f2f6;">QR Ayrışımı Hatası (MSE)</p><p style="color:#f0f2f6; font-size:30px; font-weight:bold;">{mse_qr:.2e}</p>', unsafe_allow_html=True)
#st.metric(label="...", value="...") komutu bu metin stiliyle uyumsuz olduğu için markdown kullandım.

# --- GRAFİK ---
fig = go.Figure()
# Noktalar (Elmas şeklinde)
fig.add_trace(go.Scatter(x=x_verisi, y=y_verisi, mode='markers', marker=dict(symbol='diamond', size=8, color='white', line=dict(width=1.5, color='black')), name='Veri'))
x_c = np.linspace(x_verisi.min()-0.5, x_verisi.max()+0.5, 300)
A_c = np.vander(x_c, N=derece + 1, increasing=True)

if ekk_basarili:
    # EKK Doğrusu (Mavi)
    fig.add_trace(go.Scatter(x=x_c, y=A_c @ beta_ekk, mode='lines', line=dict(color='#1f77b4', width=4), name='EKK'))
# QR Doğrusu (Kırmızı kesik)
fig.add_trace(go.Scatter(x=x_c, y=A_c @ beta_qr, mode='lines', line=dict(color='#d62728', width=4, dash='dash'), name='QR'))

# Eksen sınırlarını ayarla
f = max(y_verisi.max() - y_verisi.min(), 1)
# Arka planı şeffaf ve metinleri açık gri yap
fig.update_layout(
    yaxis_range=[y_verisi.min()-f*0.3, y_verisi.max()+f*0.3], 
    template="plotly_white", 
    height=500,
    xaxis=dict(gridcolor='rgba(240, 242, 246, 0.2)', title=dict(text='X Ekseni', font=dict(color='#f0f2f6'))),
    yaxis=dict(gridcolor='rgba(240, 242, 246, 0.2)', title=dict(text='Y', font=dict(color='#f0f2f6'))),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#f0f2f6'),
    legend=dict(font=dict(color='#f0f2f6'))
)

st.plotly_chart(fig, use_container_width=True)
