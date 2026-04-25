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
import base64, os

# ──────────────────────────────────────────────────────────────────
# SAYFA YAPILANDIRMASI
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EKK vs QR | Hesaplama Motoru",
    page_icon="∑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# ARKA PLAN & TEMA
# ──────────────────────────────────────────────────────────────────
def get_b64(path: str):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

bg_b64 = get_b64("image_13.PNG")

if bg_b64:
    bg_css = f"""
    .stApp::before {{
        content:""; position:fixed; inset:0;
        background:url("data:image/png;base64,{bg_b64}") center/cover no-repeat;
        filter:brightness(0.20); z-index:-1;
    }}
    .stApp {{ background:transparent; }}
    """
else:
    bg_css = ".stApp { background: linear-gradient(135deg,#080818 0%,#0c1828 45%,#160826 100%); }"

CSS = bg_css + """
html,body,[class*="css"]         { color:#f1f5f9!important; font-family:'Segoe UI','Inter',sans-serif; }

/* ── Hero ── */
.hero-title {
    font-size:2.5rem; font-weight:800;
    background:linear-gradient(90deg,#818cf8,#c084fc,#38bdf8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    text-align:center; letter-spacing:-.5px; margin-bottom:.1rem;
}
.hero-sub { text-align:center; color:#cbd5e1; font-size:1.02rem; margin-bottom:1.4rem; }

/* ── Glass Card ── */
.glass-card {
    background:rgba(15,20,50,0.82); border:1px solid rgba(129,140,248,.35);
    border-radius:16px; padding:1.4rem 1.6rem; margin-bottom:1.2rem;
    backdrop-filter:blur(14px);
}

/* ── Sidebar ── */
[data-testid="stSidebar"]   { background:rgba(8,8,28,.88)!important; border-right:1px solid rgba(129,140,248,.3); backdrop-filter:blur(20px); }
[data-testid="stSidebar"] * { color:#e2e8f0!important; }

/* ── Butonlar ── */
.stButton>button { background:linear-gradient(135deg,#6366f1,#a855f7)!important; color:#fff!important; border:none!important; border-radius:10px!important; font-weight:700!important; padding:.55rem 1.4rem!important; transition:opacity .2s!important; }
.stButton>button:hover { opacity:.82!important; }

/* ── Input / Select / Textarea ── */
.stSelectbox>div>div,
.stNumberInput>div>div>input,
.stTextInput>div>div>input,
.stTextArea textarea {
    background:rgba(15,20,50,.85)!important; border:1px solid rgba(129,140,248,.5)!important;
    border-radius:8px!important; color:#f1f5f9!important;
}

/* ── Slider ── */
[data-testid="stSlider"]>div>div>div { background:linear-gradient(90deg,#6366f1,#a855f7)!important; }

/* ── Tabs ── */
[data-testid="stTabs"] { background:rgba(10,12,35,.80); border-radius:12px; padding:.3rem .5rem 0; border:1px solid rgba(129,140,248,.25); }
[data-testid="stTabs"] button { color:#94a3b8!important; font-weight:600; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#c084fc!important; border-bottom:2px solid #c084fc!important; background:rgba(192,132,252,.08)!important; }

/* ── Tab İçerik (GÖRÜNÜRLÜK ARTTIRILDI) ── */
[data-testid="stTabsContent"] {
    background:rgba(8,10,38,.95)!important;
    border:1px solid rgba(129,140,248,.30);
    border-radius:0 0 12px 12px;
    padding:1.4rem!important;
}
/* Tab içindeki tüm metinleri beyaz yap */
[data-testid="stTabsContent"] p,
[data-testid="stTabsContent"] span,
[data-testid="stTabsContent"] div { color:#f1f5f9!important; }

/* ── LaTeX (GÖRÜNÜRLÜK ARTTIRILDI) ── */
.katex, .katex * { color:#f0f4ff!important; font-size:1.1em!important; }
.stLatex {
    background:rgba(8,10,40,.95)!important;
    border:1px solid rgba(129,140,248,.40)!important;
    border-radius:10px; padding:1rem 1.4rem!important; margin:.5rem 0!important;
}

/* ── Metrik ── */
[data-testid="stMetric"] { background:rgba(15,20,55,.88); border:1px solid rgba(129,140,248,.35); border-radius:12px; padding:.8rem 1.1rem; }
[data-testid="stMetricValue"] { color:#e0e7ff!important; font-size:1.3rem!important; }
[data-testid="stMetricLabel"] { color:#a5b4fc!important; }

/* ── DataFrame / DataEditor ── */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
    background:rgba(10,14,45,.92)!important; border:1px solid rgba(129,140,248,.3)!important; border-radius:10px;
}

/* ── Uyarı kutuları ── */
[data-testid="stInfo"]    { background:rgba(56,189,248,.14)!important; border-color:#38bdf8!important; color:#e0f7ff!important; }
[data-testid="stSuccess"] { background:rgba(52,211,153,.14)!important; border-color:#34d399!important; color:#d1fae5!important; }
[data-testid="stWarning"] { background:rgba(251,191,36,.14)!important; border-color:#fbbf24!important; color:#fef3c7!important; }
[data-testid="stError"]   { background:rgba(248,113,113,.17)!important; border-color:#f87171!important; color:#fee2e2!important; }
[data-testid="stInfo"] *,[data-testid="stSuccess"] *,[data-testid="stWarning"] *,[data-testid="stError"] * { color:inherit!important; }

/* ── Expander ── */
[data-testid="stExpander"] { background:rgba(12,16,42,.90); border:1px solid rgba(129,140,248,.28); border-radius:12px; }
[data-testid="stExpander"] summary { color:#c4b5fd!important; font-weight:600; }

hr { border-color:rgba(129,140,248,.25)!important; }
[data-testid="stCheckbox"] label { color:#e2e8f0!important; }
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# HAZIR VERİ SETLERİ
# ──────────────────────────────────────────────────────────────────
rng = np.random.RandomState(42)
PRESETS = {
    "🏠 Ev Fiyatları (Lineer)": {
        "x": [50,65,80,95,110,130,150,170,200,230],
        "y": [120,150,175,210,240,290,340,380,450,520],
        "desc": "Metrekare → Fiyat (bin ₺) — lineer ilişki.",
        "degree": 2,
    },
    "📡 Radar Sinyalleri (Dalgalı)": {
        "x": np.linspace(0, 4*np.pi, 18).tolist(),
        "y": (np.sin(np.linspace(0,4*np.pi,18)) + 0.3*rng.randn(18)).tolist(),
        "desc": "Zaman → Sinyal gücü — yüksek derece gerektirir.",
        "degree": 7,
    },
    "🌡️ Sıcaklık Değişimi (Mevsimsel)": {
        "x": list(range(1,13)),
        "y": [3.2,4.8,9.5,15.1,20.4,25.8,28.2,27.6,22.3,15.9,9.1,4.5],
        "desc": "Ay → Ortalama sıcaklık (°C) — periyodik örüntü.",
        "degree": 5,
    },
    "✏️ Manuel Veri Girişi": {
        "x": [1,2,3,4,5,6,7],
        "y": [2.1,4.5,9.2,16.8,26.1,37.9,51.4],
        "desc": "Aşağıdaki tablodan kendi verilerinizi giriniz.",
        "degree": 3,
    },
}


# ──────────────────────────────────────────────────────────────────
# MATEMATİK ARAÇLARI
# ──────────────────────────────────────────────────────────────────
def vandermonde(x: np.ndarray, d: int) -> np.ndarray:
    return np.vander(x, N=d+1, increasing=True)

def fit_ekk(A, y):
    AtA = A.T @ A
    kappa = np.linalg.cond(AtA)
    try:    beta = np.linalg.solve(AtA, A.T @ y)
    except: beta = np.full(A.shape[1], np.nan)
    return beta, AtA, kappa

def fit_qr(A, y):
    Q, R = np.linalg.qr(A)
    beta = np.linalg.solve(R, Q.T @ y)
    return beta, Q, R

def poly_eval(beta, x):
    return sum(beta[i] * x**i for i in range(len(beta)))

def r2(y, yh):
    tss = np.sum((y - np.mean(y))**2)
    return 1.0 - np.sum((y-yh)**2)/tss if tss > 0 else 0.0

def rss(e): return float(e @ e)

def fmt_latex(M: np.ndarray, name: str, cap: int = 5) -> str:
    r, c = M.shape
    Md = M[:cap, :cap]
    rows = r" \\ ".join(" & ".join(f"{v:.4g}" for v in row) for row in Md)
    t = rf"\mathbf{{{name}}} = \begin{{pmatrix}} {rows} \end{{pmatrix}}"
    if r > cap or c > cap:
        t += rf"\;\small\text{{(ilk {cap}\times{cap})}}"
    return t

def safe_eval(expr: str, x_arr: np.ndarray):
    ns = {"x":x_arr,"np":np,
          "sin":np.sin,"cos":np.cos,"tan":np.tan,"exp":np.exp,
          "log":np.log,"log2":np.log2,"log10":np.log10,"sqrt":np.sqrt,
          "abs":np.abs,"pi":np.pi,"e":np.e,"sinh":np.sinh,"cosh":np.cosh}
    try:
        return np.asarray(eval(compile(expr,"<s>","eval"),{"__builtins__":{}},ns),dtype=float)
    except:
        return None


# ══════════════════════════════════════════════════════════════════
# ARAYÜZ
# ══════════════════════════════════════════════════════════════════

# ── Hero ──
st.markdown(
    '<div class="hero-title">∑ EKK vs QR Ayrışımı — Hesaplama Motoru</div>'
    '<div class="hero-sub">Sayısal Kararlılık & Polinom Regresyonu · Matematik Bitirme Projesi</div>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ── Sidebar ──
with st.sidebar:
    st.markdown("## ⚙️ Kontrol Paneli")
    st.markdown("---")

    preset_name = st.selectbox("📂 Veri Seti", list(PRESETS.keys()))
    preset = PRESETS[preset_name]
    st.caption(f"ℹ️ {preset['desc']}")

    st.markdown("---")
    degree = st.slider("📐 Polinom Derecesi (d)", 1, 12, preset["degree"])

    st.markdown("---")
    st.markdown("### 🎨 Grafik")
    show_ekk       = st.checkbox("EKK Eğrisi",          value=True)
    show_qr        = st.checkbox("QR Eğrisi",           value=True)
    show_custom    = st.checkbox("Özel Denklem",         value=True)
    show_residuals = st.checkbox("Kalıntı Çizgileri",   value=False)
    show_band      = st.checkbox("QR ±σ Bölgesi",       value=False)

    st.markdown("---")
    mat_cap = st.slider("Matris boyutu (satır/sütun)", 3, 8, 5)
    st.caption("κ > 10¹²  →  EKK güvenilmez")


# ── Veri Editörü ──
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 📋 Veri Seti — Düzenlenebilir Tablo")
init_df = pd.DataFrame({"x (Bağımsız)": list(preset["x"]), "y (Bağımlı)": list(preset["y"])})
edited_df = st.data_editor(
    init_df, num_rows="dynamic", use_container_width=True,
    column_config={
        "x (Bağımsız)": st.column_config.NumberColumn(format="%.4f"),
        "y (Bağımlı)":  st.column_config.NumberColumn(format="%.4f"),
    },
    key=f"de_{preset_name}",
)
st.markdown("</div>", unsafe_allow_html=True)


# ── Özel Denklem Girişi ──
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 🧮 Kendi Denklemini Gir")
st.caption("NumPy fonksiyonları kullanılabilir: `sin(x)`,  `exp(-x)`,  `x**3 + 2*x`,  `log(x+1)`  vb.")

col_e, col_b = st.columns([4, 1])
with col_e:
    custom_expr = st.text_input("f(x) =", value="sin(x)", placeholder="örn: 0.5*x**2 - 3*x + 1")
with col_b:
    st.markdown("<br>", unsafe_allow_html=True)
    test_btn = st.button("✔ Test Et")

if custom_expr:
    _t = safe_eval(custom_expr, np.array([1.0, 2.0, 3.0]))
    if _t is not None:
        st.success(f"✅ Geçerli — f(1) = {_t[0]:.4f},  f(2) = {_t[1]:.4f},  f(3) = {_t[2]:.4f}")
    else:
        st.error("❌ Geçersiz ifade. Python/NumPy sözdizimini kontrol edin.")
        custom_expr = ""

st.markdown("</div>", unsafe_allow_html=True)


# ── Veri hazırlama ──
df_c = edited_df.dropna()
if len(df_c) < degree + 2:
    st.error(f"⚠️ d={degree} için en az {degree+2} nokta gerekli.")
    st.stop()

x = df_c.iloc[:,0].to_numpy(float)
y = df_c.iloc[:,1].to_numpy(float)
n = len(x)


# ── Hesaplamalar ──
A           = vandermonde(x, degree)
b_ekk, AtA, kappa = fit_ekk(A, y)
b_qr, Q, R  = fit_qr(A, y)

xf       = np.linspace(x.min(), x.max(), 600)
yf_ekk   = poly_eval(b_ekk, xf)
yf_qr    = poly_eval(b_qr,  xf)

yh_ekk   = poly_eval(b_ekk, x)
yh_qr    = poly_eval(b_qr,  x)
e_ekk    = y - yh_ekk
e_qr     = y - yh_qr

yf_cust  = safe_eval(custom_expr, xf) if custom_expr else None

KAPPA_T  = 1e12


# ── Metrikler ──
st.markdown("---")
m1,m2,m3,m4,m5 = st.columns(5)
with m1: st.metric("n  (Veri Sayısı)", n)
with m2: st.metric("d  (Derece)", degree)
with m3: st.metric("κ(AᵀA)", f"{kappa:.3e}")
with m4: st.metric("R²  EKK", f"{r2(y,yh_ekk):.6f}")
with m5: st.metric("R²  QR",  f"{r2(y,yh_qr):.6f}")

if kappa > KAPPA_T:
    st.error(
        f"🚨 **ILL-CONDITIONED!**  κ = {kappa:.3e}  (Eşik 10¹² aşıldı)\n\n"
        f"EKK hata tahmini ≈ {kappa*1e-16:.2e}  ·  **QR kullanımı zorunludur.**"
    )
elif kappa > 1e6:
    st.warning(f"⚡ Orta düzey ill-conditioning: κ = {kappa:.3e} — Kritik uygulamalarda QR tercih edin.")
else:
    st.success(f"✅ İyi koşullanmış sistem: κ = {kappa:.3e}")

st.markdown("---")


# ══════════════════════════════════════════════════════════════════
# MATEMATİK SEKMELERİ
# ══════════════════════════════════════════════════════════════════
st.markdown("### 🔢 Matematiksel Yapılar")

tA, tAtA, tQ, tR, tBeta, tTh = st.tabs([
    "📐 Tasarım Matrisi A",
    "🔴 AᵀA  (Normal Denk.)",
    "🟢 Q  (Ortogonal)",
    "🔵 R  (Üst Üçgen)",
    "📈 Katsayılar β",
    "📚 Teorik Arka Plan",
])

with tA:
    st.latex(r"A \in \mathbb{R}^{n \times (d+1)}, \quad A_{ij} = x_i^{\,j-1}")
    st.latex(
        r"\mathbf{A} = \begin{pmatrix}"
        r"1 & x_1 & x_1^2 & \cdots & x_1^d \\"
        r"1 & x_2 & x_2^2 & \cdots & x_2^d \\"
        r"\vdots & \vdots & \vdots & \ddots & \vdots \\"
        r"1 & x_n & x_n^2 & \cdots & x_n^d"
        r"\end{pmatrix}"
    )
    st.info(f"Boyut: **{A.shape[0]} × {A.shape[1]}**")
    st.latex(fmt_latex(A, "A", mat_cap))

with tAtA:
    st.latex(
        r"\hat{\boldsymbol{\beta}}_{\text{EKK}} = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T\mathbf{y}"
    )
    st.latex(
        r"\kappa(\mathbf{A}^T\mathbf{A}) = \bigl[\kappa(\mathbf{A})\bigr]^2"
    )
    ks = "⚠️ Aşırı Yüksek!" if kappa > KAPPA_T else "✅ Kabul Edilebilir"
    st.info(f"κ(AᵀA) = **{kappa:.4e}**  |  κ(A) ≈ **{np.sqrt(abs(kappa)):.4e}**  |  {ks}")
    st.latex(fmt_latex(AtA, r"A^TA", mat_cap))

with tQ:
    st.latex(r"\mathbf{A} = \mathbf{Q}\mathbf{R}, \quad \mathbf{Q}^T\mathbf{Q} = \mathbf{I}_{d+1}")
    st.latex(r"\hat{\boldsymbol{\beta}}_{\text{QR}} = \mathbf{R}^{-1}\mathbf{Q}^T\mathbf{y}")
    err = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]), "fro")
    st.success(f"Ortonormallik doğrulandı: ‖QᵀQ − I‖_F = **{err:.2e}**")
    st.latex(fmt_latex(Q, "Q", mat_cap))

with tR:
    st.latex(r"\mathbf{R} \in \mathbb{R}^{(d+1)\times(d+1)}, \quad R_{ij}=0 \text{ for } i>j")
    st.latex(rf"\kappa(\mathbf{{R}}) \approx {np.linalg.cond(R):.4e}")
    st.latex(fmt_latex(R, "R", mat_cap))

with tBeta:
    st.latex(r"\hat{y}(x) = \sum_{k=0}^{d} \beta_k\, x^k")
    bdf = pd.DataFrame({
        "Katsayı":    [f"β_{k}" for k in range(degree+1)],
        "EKK":        b_ekk,
        "QR":         b_qr,
        "|EKK − QR|": np.abs(b_ekk - b_qr),
    })
    st.dataframe(
        bdf.style.format({"EKK":"{:.10f}","QR":"{:.10f}","|EKK − QR|":"{:.3e}"}),
        use_container_width=True,
    )
    c1,c2 = st.columns(2)
    with c1:
        st.metric("EKK R²",  f"{r2(y,yh_ekk):.8f}")
        st.metric("EKK RSS", f"{rss(e_ekk):.4e}")
    with c2:
        st.metric("QR R²",   f"{r2(y,yh_qr):.8f}")
        st.metric("QR RSS",  f"{rss(e_qr):.4e}")

with tTh:
    st.markdown("#### Neden QR Daha Kararlı?")
    st.latex(
        r"\underbrace{(\mathbf{A}^T\mathbf{A})\hat{\beta}=\mathbf{A}^T\mathbf{y}}_{\text{Normal Denk.}}"
        r"\;\Rightarrow\; \Delta\beta \sim \kappa^2 \varepsilon_{\text{mach}}"
    )
    st.latex(
        r"\underbrace{\mathbf{A}=\mathbf{QR}\;\Rightarrow\;\mathbf{R}\hat{\beta}=\mathbf{Q}^T\mathbf{y}}_{\text{QR Ayrışımı}}"
        r"\;\Rightarrow\; \Delta\beta \sim \kappa\,\varepsilon_{\text{mach}}"
    )
    st.info(
        f"Bu projede κ ≈ **{kappa:.2e}**  →  "
        f"EKK hata büyümesi QR'a kıyasla ≈ **{max(np.sqrt(max(kappa,1)),1):.1e}×** daha büyük olabilir."
    )
    st.latex(r"\kappa(\mathbf{R})=\kappa(\mathbf{A}),\quad\kappa(\mathbf{A}^T\mathbf{A})=[\kappa(\mathbf{A})]^2")


# ══════════════════════════════════════════════════════════════════
# ANA GRAFİK
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📊 Karşılaştırmalı Regresyon Grafiği")

PBASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,14,48,.70)",
    font=dict(color="#e2e8f0", family="Segoe UI", size=13),
    legend=dict(bgcolor="rgba(8,10,30,.88)", bordercolor="rgba(129,140,248,.45)", borderwidth=1, font=dict(size=12,color="#e2e8f0")),
    margin=dict(l=55,r=25,t=55,b=40),
)
AX = dict(gridcolor="rgba(129,140,248,.15)", zerolinecolor="rgba(129,140,248,.4)", color="#cbd5e1", linecolor="rgba(129,140,248,.3)")

fig = make_subplots(rows=2, cols=1, row_heights=[.70,.30], shared_xaxes=True,
                    vertical_spacing=.07,
                    subplot_titles=["Polinom Regresyonu — EKK / QR / Özel Denklem","Kalıntı Analizi"])

# Veri noktaları
fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Gözlem",
    marker=dict(size=11,color="#f8fafc",symbol="circle",line=dict(width=2,color="#818cf8"))), row=1,col=1)

# EKK
if show_ekk:
    fig.add_trace(go.Scatter(x=xf,y=yf_ekk,mode="lines",name=f"EKK (d={degree})",
        line=dict(color="#f87171",width=2.5,dash="dash")), row=1,col=1)

# QR
if show_qr:
    if show_band:
        s = np.std(e_qr)
        fig.add_trace(go.Scatter(
            x=np.concatenate([xf,xf[::-1]]),
            y=np.concatenate([yf_qr+s,(yf_qr-s)[::-1]]),
            fill="toself",fillcolor="rgba(52,211,153,.10)",
            line=dict(color="rgba(0,0,0,0)"),name="QR ±σ"), row=1,col=1)
    fig.add_trace(go.Scatter(x=xf,y=yf_qr,mode="lines",name=f"QR (d={degree})",
        line=dict(color="#34d399",width=2.5)), row=1,col=1)

# Özel denklem
if show_custom and yf_cust is not None:
    fig.add_trace(go.Scatter(x=xf,y=yf_cust,mode="lines",name=f"f(x) = {custom_expr}",
        line=dict(color="#fbbf24",width=2,dash="dot")), row=1,col=1)

# Kalıntı çizgileri
if show_residuals:
    for xi,yi,ei,qi in zip(x,y,yh_ekk,yh_qr):
        if show_ekk:
            fig.add_trace(go.Scatter(x=[xi,xi],y=[yi,ei],mode="lines",showlegend=False,
                line=dict(color="rgba(248,113,113,.45)",width=1)), row=1,col=1)
        if show_qr:
            fig.add_trace(go.Scatter(x=[xi,xi],y=[yi,qi],mode="lines",showlegend=False,
                line=dict(color="rgba(52,211,153,.45)",width=1)), row=1,col=1)

# Kalıntı barları
if show_ekk:
    fig.add_trace(go.Bar(x=x,y=e_ekk,name="EKK Kalıntı",marker_color="rgba(248,113,113,.75)"), row=2,col=1)
if show_qr:
    fig.add_trace(go.Bar(x=x,y=e_qr,name="QR Kalıntı",marker_color="rgba(52,211,153,.75)"), row=2,col=1)

fig.add_hline(y=0,line=dict(color="rgba(200,210,255,.35)",width=1,dash="dot"),row=2,col=1)
fig.update_xaxes(**AX)
fig.update_yaxes(**AX)
fig.update_layout(height=680,**PBASE)
st.plotly_chart(fig,use_container_width=True)


# ── Derece–κ Grafiği ──
with st.expander("🔬 Derece → Koşul Sayısı İlişkisi (Kararlılık Analizi)", expanded=False):
    dr = list(range(1, min(14, n-1)))
    ks_list = [np.linalg.cond(vandermonde(x,d).T @ vandermonde(x,d)) for d in dr]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dr,y=ks_list,mode="lines+markers",name="κ(AᵀA)",
        line=dict(color="#c084fc",width=2.5),marker=dict(size=8,color="#c084fc")))
    fig2.add_hline(y=KAPPA_T,line=dict(color="#f87171",width=1.5,dash="dash"),
        annotation_text="Eşik 10¹²",annotation_font_color="#f87171")
    fig2.add_vline(x=degree,line=dict(color="#fbbf24",width=1.5,dash="dot"),
        annotation_text=f"d={degree}",annotation_font_color="#fbbf24")
    fig2.update_xaxes(title_text="Polinom Derecesi d",**AX)
    fig2.update_yaxes(title_text="κ(AᵀA) — log ölçek",type="log",**AX)
    fig2.update_layout(height=380,**PBASE)
    st.plotly_chart(fig2,use_container_width=True)


# ── Akademik Özet ──
st.markdown("---")
st.markdown("### 📄 Akademik Özet Rapor")
with st.expander("Sonuçları Göster", expanded=True):
    bd = np.linalg.norm(b_ekk - b_qr)
    st.markdown(f"""
| Metrik | EKK | QR |
|:--|--:|--:|
| R² | `{r2(y,yh_ekk):.10f}` | `{r2(y,yh_qr):.10f}` |
| RSS | `{rss(e_ekk):.6e}` | `{rss(e_qr):.6e}` |
| Max \|Kalıntı\| | `{np.max(np.abs(e_ekk)):.6e}` | `{np.max(np.abs(e_qr)):.6e}` |
| ‖β_EKK − β_QR‖₂ | `{bd:.6e}` | — |
| κ (Koşul Sayısı) | `{kappa:.6e}` | `{np.linalg.cond(R):.6e}` (R) |
| Kararlılık | {"❌ Güvenilmez" if kappa>KAPPA_T else "⚠️ Dikkat" if kappa>1e6 else "✅ İyi"} | ✅ Kararlı |
""")
    st.latex(
        r"\kappa(\mathbf{R})=\kappa(\mathbf{A}),\quad"
        r"\kappa(\mathbf{A}^T\mathbf{A})=[\kappa(\mathbf{A})]^2"
        r"\;\implies\;\text{QR her zaman daha kararlıdır.}"
    )

st.markdown("---")
st.caption(
    "🎓 Matematik Bitirme Projesi · EKK vs QR Hesaplama Motoru "
    "· Streamlit + NumPy + Plotly "
    "· IEEE 754 ε_mach ≈ 2.22×10⁻¹⁶"
)
