"""
╔══════════════════════════════════════════════════════════════════╗
║   EKK vs QR Ayrışımı — Karşılaştırmalı Hesaplama Motoru         ║
║   Matematik Bitirme Projesi  ·  Streamlit + NumPy + Plotly       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import os

# ──────────────────────────────────────────────────────────────────
# SAYFA YAPILANDIRMASI
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EKK vs QR Ayrışımı | Hesaplama Motoru",
    page_icon="∑",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────
# ARKA PLAN & TEMA (CSS)
# ──────────────────────────────────────────────────────────────────
def get_base64_image(image_path: str) -> str | None:
    """Görseli base64'e çevirir; dosya yoksa None döner."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


BG_IMAGE_PATH = "image_13.PNG"
bg_b64 = get_base64_image(BG_IMAGE_PATH)

if bg_b64:
    bg_css = f"""
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: url("data:image/png;base64,{bg_b64}") center/cover no-repeat;
        filter: brightness(0.20);
        z-index: -1;
    }}
    .stApp {{ background: transparent; }}
    """
else:
    # Görsel yoksa koyu gradyan arka plan
    bg_css = """
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 40%, #1a0a2e 100%);
    }
    """

THEME_CSS = bg_css + """
/* ── Genel Metin ─────────────────────────────────────────────── */
html, body, [class*="css"] {
    color: #e8eaf6 !important;
    font-family: 'Segoe UI', 'Inter', sans-serif;
}

/* ── Başlık Gradyanı ──────────────────────────────────────────── */
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #7c83fd, #c084fc, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
}
.hero-sub {
    text-align: center;
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 1.8rem;
}

/* ── Kartlar ──────────────────────────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(12px);
}

/* ── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(10, 10, 30, 0.75) !important;
    border-right: 1px solid rgba(124,131,253,0.3);
    backdrop-filter: blur(16px);
}

/* ── Butonlar ─────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #7c83fd, #c084fc) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Sliderlar & Selectboxlar ─────────────────────────────────── */
[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, #7c83fd, #c084fc) !important;
}
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(124,131,253,0.4) !important;
    border-radius: 8px !important;
    color: #e8eaf6 !important;
}

/* ── Tabs ─────────────────────────────────────────────────────── */
[data-testid="stTabs"] button {
    color: #94a3b8 !important;
    border-radius: 8px 8px 0 0 !important;
    font-weight: 600;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #c084fc !important;
    border-bottom: 2px solid #c084fc !important;
}

/* ── Metrik Kutuları ──────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(124,131,253,0.25);
    border-radius: 12px;
    padding: 0.8rem 1rem;
}

/* ── Data Editor ──────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px;
}

/* ── Divider ──────────────────────────────────────────────────── */
hr { border-color: rgba(124,131,253,0.25) !important; }

/* ── LaTeX ────────────────────────────────────────────────────── */
.katex { color: #c8d6f5 !important; font-size: 1.05em !important; }

/* ── Expander ─────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
}
"""

st.markdown(f"<style>{THEME_CSS}</style>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# HAZIR VERİ SETLERİ
# ──────────────────────────────────────────────────────────────────
PRESET_DATASETS: dict[str, dict] = {
    "🏠 Ev Fiyatları (Lineer)": {
        "x": [50, 65, 80, 95, 110, 130, 150, 170, 200, 230],
        "y": [120, 150, 175, 210, 240, 290, 340, 380, 450, 520],
        "desc": "Metrekare → Fiyat (bin ₺) — lineer ilişki beklenir.",
        "degree": 2,
    },
    "📡 Radar Sinyalleri (Dalgalı)": {
        "x": np.linspace(0, 4 * np.pi, 18).tolist(),
        "y": (np.sin(np.linspace(0, 4 * np.pi, 18)) + 0.3 * np.random.RandomState(42).randn(18)).tolist(),
        "desc": "Zaman → Sinyal gücü — yüksek dereceli polinom gerektirir.",
        "degree": 7,
    },
    "🌡️ Sıcaklık Değişimi (Mevsimsel)": {
        "x": list(range(1, 13)),
        "y": [3.2, 4.8, 9.5, 15.1, 20.4, 25.8, 28.2, 27.6, 22.3, 15.9, 9.1, 4.5],
        "desc": "Ay → Ortalama sıcaklık (°C) — periyodik örüntü.",
        "degree": 5,
    },
    "⚡ Özel Veri (Manuel Giriş)": {
        "x": [1, 2, 3, 4, 5, 6, 7],
        "y": [2.1, 4.5, 9.2, 16.8, 26.1, 37.9, 51.4],
        "desc": "Kendi verilerinizi aşağıdaki tablodan giriniz.",
        "degree": 3,
    },
}


# ──────────────────────────────────────────────────────────────────
# TEMEL MATEMATİKSEL FONKSİYONLAR
# ──────────────────────────────────────────────────────────────────
def build_vandermonde(x: np.ndarray, degree: int) -> np.ndarray:
    """Vandermonde (Tasarım) Matrisi A ∈ ℝ^{n×(d+1)}."""
    return np.vander(x, N=degree + 1, increasing=True)


def fit_ekk_normal(A: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Klasik EKK: β = (AᵀA)⁻¹ Aᵀy  —  Normal Denklemler yoluyla.
    Döndürür: (β katsayıları, koşul sayısı κ(AᵀA))
    """
    AtA = A.T @ A
    kappa = np.linalg.cond(AtA)
    try:
        beta = np.linalg.solve(AtA, A.T @ y)
    except np.linalg.LinAlgError:
        beta = np.full(A.shape[1], np.nan)
    return beta, kappa


def fit_qr(A: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    QR Ayrışımı: A = QR  →  β = R⁻¹ Qᵀy
    Döndürür: (β, Q, R)
    """
    Q, R = np.linalg.qr(A)
    beta = np.linalg.solve(R, Q.T @ y)
    return beta, Q, R


def residuals(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """Kalıntı vektörü: e = y − ŷ"""
    return y - y_hat


def rss(e: np.ndarray) -> float:
    """Artık Kareler Toplamı (RSS): eᵀe"""
    return float(e @ e)


def r_squared(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Determinasyon katsayısı: R² = 1 − RSS/TSS"""
    tss = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - rss(residuals(y, y_hat)) / tss if tss > 0 else 0.0


def poly_eval(beta: np.ndarray, x_fine: np.ndarray) -> np.ndarray:
    """β katsayıları ile polinom değerlendirir: p(x) = Σ βᵢ xⁱ"""
    return sum(beta[i] * x_fine ** i for i in range(len(beta)))


def format_matrix_latex(M: np.ndarray, name: str, max_size: int = 6) -> str:
    """NumPy matrisini LaTeX pmatrix formatına çevirir."""
    r, c = M.shape
    if r > max_size or c > max_size:
        M = M[:max_size, :max_size]
        truncated = True
    else:
        truncated = False

    rows_str = r" \\ ".join(
        " & ".join(f"{v:.4f}" for v in row) for row in M
    )
    latex = rf"\mathbf{{{name}}} = \begin{{pmatrix}} {rows_str} \end{{pmatrix}}"
    if truncated:
        latex += rf"\quad \text{{(ilk {max_size}\times{max_size} gösteriliyor)}}"
    return latex


# ──────────────────────────────────────────────────────────────────
# HERO BAŞLIĞI
# ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-title">∑ EKK vs QR Ayrışımı — Hesaplama Motoru</div>
    <div class="hero-sub">
        Sayısal Kararlılık & Polinom Regresyonu · Matematik Bitirme Projesi
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# ──────────────────────────────────────────────────────────────────
# KENAR ÇUBUĞU (Kontrol Paneli)
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Kontrol Paneli")
    st.markdown("---")

    # Veri seti seçimi
    preset_name = st.selectbox(
        "📂 Veri Seti Seçin",
        options=list(PRESET_DATASETS.keys()),
        help="Hazır veri setlerinden birini seçin veya manuel giriş yapın.",
    )
    preset = PRESET_DATASETS[preset_name]
    st.caption(f"ℹ️ {preset['desc']}")

    st.markdown("---")

    # Polinom derecesi
    degree = st.slider(
        "📐 Polinom Derecesi (d)",
        min_value=1,
        max_value=12,
        value=preset["degree"],
        help="Yüksek derece → ill-conditioning riski artar.",
    )

    st.markdown("---")

    # Görsel seçenekler
    st.markdown("### 🎨 Grafik Seçenekleri")
    show_ekk = st.checkbox("EKK Eğrisi", value=True)
    show_qr = st.checkbox("QR Eğrisi", value=True)
    show_residuals = st.checkbox("Kalıntı Çizgileri", value=False)
    show_confidence = st.checkbox("Interpolasyon Bölgesi", value=False)

    st.markdown("---")
    st.markdown("### 📊 Matris Boyut Sınırı")
    max_matrix_display = st.slider("Gösterilecek satır/sütun sayısı", 3, 8, 6)

    st.markdown("---")
    st.caption("**Not:** Condition Number > 10¹² ise EKK güvenilmez.")


# ──────────────────────────────────────────────────────────────────
# VERİ EDITÖRÜ
# ──────────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 📋 Veri Seti — Düzenlenebilir Tablo")
st.caption("Satır ekleyip silebilir, değerleri değiştirebilirsiniz.")

init_df = pd.DataFrame(
    {"x (Bağımsız Değişken)": preset["x"], "y (Bağımlı Değişken)": preset["y"]}
)

edited_df = st.data_editor(
    init_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "x (Bağımsız Değişken)": st.column_config.NumberColumn(format="%.4f"),
        "y (Bağımlı Değişken)": st.column_config.NumberColumn(format="%.4f"),
    },
    key=f"editor_{preset_name}",
)
st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# VERİ HAZIRLIĞI
# ──────────────────────────────────────────────────────────────────
df_clean = edited_df.dropna()
if len(df_clean) < degree + 2:
    st.error(
        f"⚠️ En az **{degree + 2}** veri noktası gerekli "
        f"(d={degree} için). Lütfen daha fazla veri ekleyin veya dereceyi düşürün."
    )
    st.stop()

x_data = df_clean.iloc[:, 0].to_numpy(dtype=float)
y_data = df_clean.iloc[:, 1].to_numpy(dtype=float)
n = len(x_data)

# ──────────────────────────────────────────────────────────────────
# HESAPLAMALAR
# ──────────────────────────────────────────────────────────────────
A = build_vandermonde(x_data, degree)
AtA = A.T @ A

beta_ekk, kappa = fit_ekk_normal(A, y_data)
beta_qr, Q_mat, R_mat = fit_qr(A, y_data)

x_fine = np.linspace(x_data.min(), x_data.max(), 500)
y_ekk_fine = poly_eval(beta_ekk, x_fine)
y_qr_fine = poly_eval(beta_qr, x_fine)

y_hat_ekk = poly_eval(beta_ekk, x_data)
y_hat_qr = poly_eval(beta_qr, x_data)

e_ekk = residuals(y_data, y_hat_ekk)
e_qr = residuals(y_data, y_hat_qr)

r2_ekk = r_squared(y_data, y_hat_ekk)
r2_qr = r_squared(y_data, y_hat_qr)

# ──────────────────────────────────────────────────────────────────
# CONDITION NUMBER UYARISI
# ──────────────────────────────────────────────────────────────────
KAPPA_THRESHOLD = 1e12
st.markdown("---")
col_m1, col_m2, col_m3, col_m4 = st.columns(4)

with col_m1:
    st.metric("📏 Veri Noktası Sayısı (n)", n)
with col_m2:
    st.metric("📐 Polinom Derecesi (d)", degree)
with col_m3:
    kappa_str = f"{kappa:.3e}" if kappa < 1e15 else "∞ (Tekil!)"
    st.metric("⚠️ Koşul Sayısı κ(AᵀA)", kappa_str)
with col_m4:
    st.metric("🏆 QR — R²", f"{r2_qr:.6f}")

if kappa > KAPPA_THRESHOLD:
    st.error(
        f"""
        🚨 **ILL-CONDITIONED SİSTEM TESPİT EDİLDİ**

        Koşul Sayısı κ(AᵀA) = {kappa:.3e} → Eşik değer 10¹² aşıldı!

        **Sonuç:** Normal Denklem sistemi (EKK) sayısal olarak kararsızdır.
        Makine epsilon'u (~10⁻¹⁶) ile çarpıldığında hata büyümesi **{kappa * 1e-16:.2e}** mertebeye ulaşabilir.
        **QR Ayrışımı kullanılması akademik olarak zorunludur.**
        """
    )
elif kappa > 1e6:
    st.warning(
        f"⚡ Orta düzey ill-conditioning: κ = {kappa:.3e}. "
        "Yüksek hassasiyet gerektiren uygulamalarda QR tercih edilmeli."
    )
else:
    st.success(f"✅ İyi koşullanmış sistem: κ = {kappa:.3e}. Her iki yöntem de güvenilir.")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────
# MATEMATİK SEKMELERİ
# ──────────────────────────────────────────────────────────────────
st.markdown("### 🔢 Matematiksel Yapılar")

tab_A, tab_AtA, tab_Q, tab_R, tab_beta, tab_theory = st.tabs(
    ["📐 Tasarım Matrisi A", "🔴 AᵀA (Normal)", "🟢 Q (Ortogonal)", "🔵 R (Üst Üçgen)", "📈 Katsayılar β", "📚 Teorik Arka Plan"]
)

with tab_A:
    st.latex(
        r"""
        \mathbf{A} \in \mathbb{R}^{n \times (d+1)}, \quad
        A_{ij} = x_i^{j-1}, \quad i=1,\ldots,n,\; j=1,\ldots,d+1
        """
    )
    st.latex(
        r"""
        \mathbf{A} = \begin{pmatrix}
            1 & x_1 & x_1^2 & \cdots & x_1^d \\
            1 & x_2 & x_2^2 & \cdots & x_2^d \\
            \vdots & \vdots & \vdots & \ddots & \vdots \\
            1 & x_n & x_n^2 & \cdots & x_n^d
        \end{pmatrix}
        """
    )
    st.markdown(f"**Matris boyutu:** {A.shape[0]} × {A.shape[1]}")
    st.latex(format_matrix_latex(A, "A", max_matrix_display))

with tab_AtA:
    st.latex(
        r"""
        \mathbf{A}^T\mathbf{A} \in \mathbb{R}^{(d+1)\times(d+1)}, \quad
        \hat{\boldsymbol{\beta}}_{\text{EKK}} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{y}
        """
    )
    st.latex(
        r"""
        \kappa(\mathbf{A}^T\mathbf{A}) = \frac{\sigma_{\max}(\mathbf{A}^T\mathbf{A})}{\sigma_{\min}(\mathbf{A}^T\mathbf{A})}
        = \left[\kappa(\mathbf{A})\right]^2
        """
    )
    st.info(
        f"κ(AᵀA) = {kappa:.4e}  →  "
        f"κ(A) ≈ {np.sqrt(kappa):.4e}  "
        f"{'⚠️ Dikkat: Aşırı yüksek!' if kappa > KAPPA_THRESHOLD else '✅ Kabul edilebilir'}"
    )
    st.latex(format_matrix_latex(AtA, r"A^TA", max_matrix_display))

with tab_Q:
    st.latex(
        r"""
        \mathbf{A} = \mathbf{Q}\mathbf{R}, \quad
        \mathbf{Q} \in \mathbb{R}^{n \times (d+1)}, \quad
        \mathbf{Q}^T\mathbf{Q} = \mathbf{I}_{d+1}
        """
    )
    st.latex(
        r"""
        \hat{\boldsymbol{\beta}}_{\text{QR}} = \mathbf{R}^{-1}\mathbf{Q}^T\mathbf{y}
        \quad \Longleftarrow \quad
        \kappa(\mathbf{R}) = \kappa(\mathbf{A}) \ll \kappa(\mathbf{A}^T\mathbf{A})
        """
    )
    # Q'nun ortonormalliğini doğrula
    QtQ = Q_mat.T @ Q_mat
    ortho_err = np.linalg.norm(QtQ - np.eye(Q_mat.shape[1]), "fro")
    st.success(f"✅ Ortonormallik hatası ‖QᵀQ − I‖_F = {ortho_err:.2e}")
    st.latex(format_matrix_latex(Q_mat, "Q", max_matrix_display))

with tab_R:
    st.latex(
        r"""
        \mathbf{R} \in \mathbb{R}^{(d+1)\times(d+1)}, \quad
        R_{ij} = 0 \text{ için } i > j \quad \text{(üst üçgen)}
        """
    )
    st.latex(
        r"""
        \text{RSS}_{\text{QR}} = \|\mathbf{y} - \mathbf{A}\hat{\boldsymbol{\beta}}_{\text{QR}}\|_2^2
        = \|\mathbf{Q}^T\mathbf{y} - \mathbf{R}\hat{\boldsymbol{\beta}}_{\text{QR}}\|_2^2
        """
    )
    st.latex(format_matrix_latex(R_mat, "R", max_matrix_display))

with tab_beta:
    st.latex(
        r"""
        \hat{\boldsymbol{\beta}} = [\beta_0, \beta_1, \ldots, \beta_d]^T, \quad
        \hat{y}(x) = \sum_{k=0}^{d} \beta_k x^k
        """
    )
    beta_df = pd.DataFrame(
        {
            "Katsayı": [f"β_{k}" for k in range(degree + 1)],
            "EKK Değeri": beta_ekk,
            "QR Değeri": beta_qr,
            "Fark |EKK − QR|": np.abs(beta_ekk - beta_qr),
        }
    )
    st.dataframe(
        beta_df.style.format({"EKK Değeri": "{:.8f}", "QR Değeri": "{:.8f}", "Fark |EKK − QR|": "{:.2e}"}),
        use_container_width=True,
    )
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.metric("EKK — R²", f"{r2_ekk:.6f}")
        st.metric("EKK — RSS", f"{rss(e_ekk):.4e}")
    with col_r2:
        st.metric("QR — R²", f"{r2_qr:.6f}")
        st.metric("QR — RSS", f"{rss(e_qr):.4e}")

with tab_theory:
    st.markdown("#### 📖 Teorik Karşılaştırma")
    st.latex(
        r"""
        \underbrace{\hat{\boldsymbol{\beta}}_{\text{EKK}} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{y}}_{\text{Normal Denklemler}} \quad \xrightarrow{\kappa^2 \text{ hata büyümesi}} \quad \text{Sayısal Kararsızlık}
        """
    )
    st.latex(
        r"""
        \underbrace{\mathbf{A} = \mathbf{Q}\mathbf{R} \;\Rightarrow\; \hat{\boldsymbol{\beta}}_{\text{QR}} = \mathbf{R}^{-1}\mathbf{Q}^T\mathbf{y}}_{\text{Householder Yansımaları}} \quad \xrightarrow{\kappa \text{ hata büyümesi}} \quad \text{Sayısal Kararlılık}
        """
    )
    st.info(
        "**Temel sonuç:** EKK'da hata büyümesi koşul sayısının **karesiyle** orantılıyken, "
        "QR ayrışımında yalnızca koşul sayısıyla orantılıdır. "
        f"Bu projede κ ≈ {kappa:.2e} → EKK hatası QR hatasına kıyasla yaklaşık "
        f"{kappa * 1e-16 / max(np.sqrt(kappa) * 1e-16, 1e-30):.1f}× daha büyük olabilir."
    )


# ──────────────────────────────────────────────────────────────────
# ANA GRAFİK (Plotly)
# ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Karşılaştırmalı Regresyon Grafiği")

fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.72, 0.28],
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=["Polinom Regresyonu: EKK vs QR Ayrışımı", "Kalıntı (Artık) Analizi"],
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.04)",
    font=dict(color="#e8eaf6", family="Segoe UI"),
    margin=dict(l=50, r=30, t=60, b=40),
    legend=dict(
        bgcolor="rgba(10,10,30,0.7)",
        bordercolor="rgba(124,131,253,0.4)",
        borderwidth=1,
        font=dict(size=12),
    ),
    xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)"),
    xaxis2=dict(gridcolor="rgba(255,255,255,0.08)", title_text="x"),
    yaxis2=dict(gridcolor="rgba(255,255,255,0.08)", title_text="Kalıntı eᵢ"),
)

# Veri noktaları
fig.add_trace(
    go.Scatter(
        x=x_data, y=y_data,
        mode="markers",
        name="Gözlem Verisi",
        marker=dict(size=10, color="#f8fafc", symbol="circle",
                    line=dict(width=2, color="#7c83fd")),
    ),
    row=1, col=1,
)

# EKK eğrisi
if show_ekk:
    fig.add_trace(
        go.Scatter(
            x=x_fine, y=y_ekk_fine,
            mode="lines",
            name=f"EKK (d={degree})",
            line=dict(color="#f87171", width=2.5, dash="dash"),
        ),
        row=1, col=1,
    )

# QR eğrisi
if show_qr:
    fig.add_trace(
        go.Scatter(
            x=x_fine, y=y_qr_fine,
            mode="lines",
            name=f"QR (d={degree})",
            line=dict(color="#34d399", width=2.5),
        ),
        row=1, col=1,
    )

# İnterpolasyon bölgesi (QR ±std)
if show_confidence and show_qr:
    std_qr = np.std(e_qr)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_fine, x_fine[::-1]]),
            y=np.concatenate([y_qr_fine + std_qr, (y_qr_fine - std_qr)[::-1]]),
            fill="toself",
            fillcolor="rgba(52,211,153,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="QR ±σ Bölgesi",
            showlegend=True,
        ),
        row=1, col=1,
    )

# Kalıntı çizgileri
if show_residuals:
    for xi, yi, yh_ekk, yh_qr in zip(x_data, y_data, y_hat_ekk, y_hat_qr):
        if show_ekk:
            fig.add_trace(
                go.Scatter(
                    x=[xi, xi], y=[yi, yh_ekk],
                    mode="lines",
                    line=dict(color="rgba(248,113,113,0.5)", width=1, dash="dot"),
                    showlegend=False,
                ),
                row=1, col=1,
            )
        if show_qr:
            fig.add_trace(
                go.Scatter(
                    x=[xi, xi], y=[yi, yh_qr],
                    mode="lines",
                    line=dict(color="rgba(52,211,153,0.5)", width=1, dash="dot"),
                    showlegend=False,
                ),
                row=1, col=1,
            )

# Kalıntı alt grafiği
if show_ekk:
    fig.add_trace(
        go.Bar(
            x=x_data, y=e_ekk,
            name="EKK Kalıntıları",
            marker_color="rgba(248,113,113,0.7)",
        ),
        row=2, col=1,
    )
if show_qr:
    fig.add_trace(
        go.Bar(
            x=x_data, y=e_qr,
            name="QR Kalıntıları",
            marker_color="rgba(52,211,153,0.7)",
        ),
        row=2, col=1,
    )

# Sıfır çizgisi
fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"), row=2, col=1)

fig.update_layout(height=680, **PLOTLY_LAYOUT)
st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# CONDITION NUMBER KARŞILAŞTIRMA GRAFİĞİ
# ──────────────────────────────────────────────────────────────────
with st.expander("🔬 Derece–Koşul Sayısı İlişkisi (Kararlılık Analizi)", expanded=False):
    st.markdown(
        "Polinom derecesi arttıkça κ(AᵀA) üssel olarak büyür. "
        "Bu grafik, hangi dereceden itibaren EKK'nın güvenilmezleştiğini gösterir."
    )
    degrees_range = list(range(1, min(14, n - 1)))
    kappas = []
    for d in degrees_range:
        A_tmp = build_vandermonde(x_data, d)
        AtA_tmp = A_tmp.T @ A_tmp
        k = np.linalg.cond(AtA_tmp)
        kappas.append(k)

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=degrees_range, y=kappas,
            mode="lines+markers",
            name="κ(AᵀA)",
            line=dict(color="#c084fc", width=2.5),
            marker=dict(size=8, color="#c084fc"),
        )
    )
    fig2.add_hline(
        y=KAPPA_THRESHOLD,
        line=dict(color="#f87171", width=1.5, dash="dash"),
        annotation_text="Güvenilirlik Eşiği 10¹²",
        annotation_font_color="#f87171",
    )
    fig2.add_vline(
        x=degree,
        line=dict(color="#fbbf24", width=1.5, dash="dot"),
        annotation_text=f"Seçili d={degree}",
        annotation_font_color="#fbbf24",
    )
    fig2.update_layout(
        yaxis_type="log",
        yaxis_title="κ(AᵀA) — Logaritmik Ölçek",
        xaxis_title="Polinom Derecesi d",
        height=380,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig2, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# ÖZET RAPOR
# ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📄 Akademik Özet Rapor")
with st.expander("Sonuçları Göster / Gizle", expanded=True):
    st.markdown(
        f"""
| Metrik | EKK (Normal Denklem) | QR Ayrışımı |
|---|---|---|
| R² | `{r2_ekk:.8f}` | `{r2_qr:.8f}` |
| RSS | `{rss(e_ekk):.4e}` | `{rss(e_qr):.4e}` |
| Max \|Kalıntı\| | `{np.max(np.abs(e_ekk)):.4e}` | `{np.max(np.abs(e_qr)):.4e}` |
| Katsayı Farkı ‖β_EKK − β_QR‖₂ | `{np.linalg.norm(beta_ekk - beta_qr):.4e}` | — |
| Koşul Sayısı κ(AᵀA) | `{kappa:.4e}` | `{np.linalg.cond(R_mat):.4e}` (R üzerinden) |
| Hesaplama Stabilitesi | {"❌ Güvenilmez" if kappa > KAPPA_THRESHOLD else "⚠️ Dikkatli Ol" if kappa > 1e6 else "✅ İyi"} | ✅ Kararlı |
        """
    )
    st.latex(
        r"""
        \text{Sonuç: } \kappa(\mathbf{R}) = \kappa(\mathbf{A}),\quad
        \kappa(\mathbf{A}^T\mathbf{A}) = [\kappa(\mathbf{A})]^2
        \implies \text{QR Ayrışımı her zaman daha kararlıdır.}
        """
    )

st.markdown("---")
st.caption(
    "🎓 Matematik Bitirme Projesi · EKK vs QR Karşılaştırmalı Hesaplama Motoru  "
    "· Streamlit + NumPy + Plotly  "
    "· Tüm hesaplamalar IEEE 754 çift duyarlıklı aritmetik ile gerçekleştirilmiştir."
)
