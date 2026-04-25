import streamlit as st
import numpy as np
import re

# --- Sayfa Yapılandırması ---
st.set_page_config(page_title="EKK ve QR Analizi | Synapse 5", layout="wide")

# --- Başlık ---
st.title("📐 En Küçük Kareler (EKK) ve QR Ayrışımı")
st.markdown("Dinamik Regresyon Hesaplama Motoru")
st.divider()

# --- Veri Giriş Yöntemi Seçimi ---
st.header("1. Veri Hazırlama")
veri_modu = st.radio(
    "Veri giriş yöntemini seçiniz:",
    ["📐 Matematiksel Fonksiyon Yaz", "🔢 Noktaları Manuel Belirle"],
    horizontal=True
)

x_verisi = np.array([])
y_verisi = np.array([])

# --- MOD 1: FONKSİYON GİRİŞİ ---
if veri_modu == "📐 Matematiksel Fonksiyon Yaz":
    with st.container(border=True):
        st.markdown("### Fonksiyon Tanımlama")
        fonksiyon_metni = st.text_input("f(x) formülünü yazın:", value="np.sin(x) + x**2")
        st.caption("İpucu: 'np.sin(x)', 'x**2', 'np.exp(x)' gibi ifadeler kullanabilirsiniz.")
        
        col_n, col_range = st.columns(2)
        with col_n:
            nokta_sayisi = st.number_input("Oluşturulacak Nokta Sayısı:", 5, 500, 50)
        with col_range:
            aralik = st.slider("X Ekseni Aralığı:", -10.0, 10.0, (-5.0, 5.0))
        
        # Matematiksel hesaplama
        x_verisi = np.linspace(aralik[0], aralik[1], nokta_sayisi)
        try:
            # Kullanıcının metnini güvenli bir şekilde değerlendiriyoruz
            y_verisi = eval(fonksiyon_metni, {"np": np, "x": x_verisi})
        except Exception as e:
            st.error(f"Denklem hatası: {e}")
            y_verisi = np.zeros(nokta_sayisi)

# --- MOD 2: MANUEL NOKTA GİRİŞİ ---
else:
    with st.container(border=True):
        st.markdown("### Manuel Nokta Girişi")
        n_manuel = st.number_input("Kaç adet nokta gireceksiniz?", 2, 20, 5)
        
        # Yan yana dikey barlar (slider) oluşturarak interaktif giriş sağlıyoruz
        st.write("Noktaların Y değerlerini ayarlayın (X değerleri 0'dan başlar):")
        y_list = []
        kolonlar = st.columns(n_manuel)
        for i in range(n_manuel):
            with kolonlar[i]:
                y_val = st.slider(f"X={i}", -10.0, 20.0, float(i*2), key=f"sl_{i}")
                y_list.append(y_val)
        
        x_verisi = np.arange(n_manuel, dtype=float)
        y_verisi = np.array(y_list)

# --- Kontrol ve Onay ---
if len(x_verisi) > 0:
    st.info(f"Sistem hazır: {len(x_verisi)} adet veri noktası oluşturuldu.")
