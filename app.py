import streamlit as st
import pandas as pd
import numpy as np

# --- Sayfa Yapılandırması ---
# 'wide' layout ile matematiksel matrisler ve grafikler için geniş bir alan açıyoruz.
st.set_page_config(page_title="EKK ve QR Analizi | Synapse 5", layout="wide")

# --- Başlık ve Giriş ---
st.title("📐 En Küçük Kareler (EKK) ve QR Ayrışımı")
st.markdown("**Synapse 5 Proje Grubu** | Dinamik Regresyon Hesaplama Motoru")
st.divider()

# --- Adım 1: Veri Girişi ---
st.header("1. Veri Noktalarının Girişi")
st.write("Analiz edilecek X ve Y veri noktalarını aşağıdaki tablodan düzenleyebilirsiniz. Tablonun altına tıklayarak yeni satırlar ekleyebilirsiniz.")

# Sistemin boş kalmaması için varsayılan bir başlangıç veri seti oluşturuyoruz
varsayilan_veri = pd.DataFrame({
    "X": [1.0, 2.0, 3.0, 4.0, 5.0],
    "Y": [2.1, 4.0, 6.2, 8.1, 10.5]
})

# st.data_editor ile kullanıcıya tabloyu düzenleme yetkisi veriyoruz
duzenlenen_veri = st.data_editor(
    varsayilan_veri, 
    num_rows="dynamic", # Dinamik satır ekleme özelliği
    use_container_width=True # Tablonun ekrana şık bir şekilde yayılması için
)

# Tablodaki verileri, ileride yapacağımız matris işlemleri için arka planda Numpy dizilerine çekiyoruz
x_noktalari = duzenlenen_veri["X"].to_numpy()
y_noktalari = duzenlenen_veri["Y"].to_numpy()

# İşlemin başarılı olduğunu gösteren küçük bir bildirim
st.success(f"Sisteme {len(x_noktalari)} adet veri noktası başarıyla yüklendi ve analize hazır!")
