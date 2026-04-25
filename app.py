"""
EKK vs QR Ayrışımı — Karşılaştırmalı Hesaplama Motoru
Matematik Bitirme Projesi · Streamlit + NumPy + Plotly
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64, os

# ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="EKK vs QR", page_icon="∑", layout="wide", initial_sidebar_state="expanded")

# ──────────────────────────────────────────────────────────────────
# TEMA
# ──────────────────────────────────────────────────────────────────
def get_b64(path):
    if os.path.exists(path):
        with open(path,"rb") as f: return base64.b64encode(f.read()).decode()
    return None

bg = get_b64("image_13.PNG")
bg_css = (
    f".stApp::before{{content:'';position:fixed;inset:0;"
    f"background:url('data:image/png;base64,{bg}') center/cover no-repeat;"
    f"filter:brightness(0.18);z-index:-1;}}.stApp{{background:transparent;}}"
    if bg else
    ".stApp{background:linear-gradient(135deg,#06080f 0%,#0b1220 50%,#110820 100%);}"
)

st.markdown(f"""<style>
{bg_css}

/* Base */
html,body,[class*="css"] {{ color:#f1f5f9!important; font-family:'Segoe UI',sans-serif; }}

/* Hero */
.hero {{ text-align:center; padding:1.2rem 0 0.4rem; }}
.hero h1 {{
    font-size:2.2rem; font-weight:800; margin:0;
    background:linear-gradient(90deg,#818cf8,#c084fc,#38bdf8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}}
.hero p {{ color:#94a3b8; font-size:.95rem; margin:.3rem 0 0; }}

/* Section header */
.sec-header {{
    font-size:1.05rem; font-weight:700; color:#c4b5fd;
    border-left:3px solid #7c3aed; padding-left:.6rem;
    margin:1.4rem 0 .8rem;
}}

/* Card */
.card {{
    background:rgba(12,15,40,.88); border:1px solid rgba(99,102,241,.3);
    border-radius:12px; padding:1.1rem 1.3rem; margin-bottom:.9rem;
}}

/* Sidebar */
[data-testid="stSidebar"] {{ background:rgba(6,8,20,.92)!important; border-right:1px solid rgba(99,102,241,.25); }}
[data-testid="stSidebar"] * {{ color:#e2e8f0!important; }}
[data-testid="stSidebar"] .stMarkdown h2 {{ color:#a5b4fc!important; font-size:1rem!important; }}

/* Butonlar */
.stButton>button {{
    background:linear-gradient(135deg,#4f46e5,#7c3aed)!important;
    color:#fff!important; border:none!important; border-radius:8px!important;
    font-weight:600!important; width:100%;
}}

/* Inputs */
.stSelectbox>div>div,
.stTextInput>div>div>input,
.stTextArea textarea {{
    background:rgba(12,15,40,.90)!important;
    border:1px solid rgba(99,102,241,.45)!important;
    border-radius:7px!important; color:#f1f5f9!important;
}}

/* Slider */
[data-testid="stSlider"]>div>div>div {{ background:linear-gradient(90deg,#4f46e5,#7c3aed)!important; }}

/* Tabs */
[data-testid="stTabs"] {{
    background:rgba(10,12,32,.85); border-radius:10px;
    border:1px solid rgba(99,102,241,.22); padding:.25rem .4rem 0;
}}
[data-testid="stTabs"] button {{ color:#64748b!important; font-weight:600; font-size:.88rem; }}
[data-testid="stTabs"] button[aria-selected="true"] {{ color:#a78bfa!important; border-bottom:2px solid #a78bfa!important; }}
[data-testid="stTabsContent"] {{
    background:rgba(8,10,30,.96)!important;
    border:1px solid rgba(99,102,241,.22);
    border-radius:0 0 10px 10px; padding:1.1rem!important;
}}
[data-testid="stTabsContent"] * {{ color:#e2e8f0!important; }}

/* LaTeX */
.katex,.katex * {{ color:#e0e7ff!important; font-size:.98em!important; }}
.stLatex {{
    background:rgba(15,18,50,.95)!important;
    border:1px solid rgba(99,102,241,.35)!important;
    border-radius:8px; padding:.75rem 1rem!important; margin:.35rem 0!important;
}}

/* Metrik */
[data-testid="stMetric"] {{
    background:rgba(12,15,42,.90); border:1px solid rgba(99,102,241,.30);
    border-radius:10px; padding:.7rem 1rem;
}}
[data-testid="stMetricValue"] {{ color:#c7d2fe!important; font-size:1.15rem!important; font-weight:700!important; }}
[data-testid="stMetricLabel"] {{ color:#818cf8!important; font-size:.78rem!important; }}

/* Alerts */
[data-testid="stInfo"]    {{ background:rgba(56,189,248,.10)!important; border-color:#38bdf8!important; }}
[data-testid="stSuccess"] {{ background:rgba(52,211,153,.10)!important; border-color:#34d399!important; }}
[data-testid="stWarning"] {{ background:rgba(251,191,36,.10)!important; border-color:#fbbf24!important; }}
[data-testid="stError"]   {{ background:rgba(248,113,113,.13)!important; border-color:#f87171!important; }}
[data-testid="stInfo"] *,[data-testid="stSuccess"] *,[data-testid="stWarning"] *,[data-testid="stError"] * {{ color:inherit!important; }}

/* Expander */
[data-testid="stExpander"] {{
    background:rgba(10,12,34,.90); border:1px solid rgba(99,102,241,.22); border-radius:10px;
}}
[data-testid="stExpander"] summary {{ color:#a78bfa!important; font-weight:600; font-size:.93rem; }}

/* Matris tablosu */
.matrix-wrap {{
    overflow-x:auto; margin:.3rem 0;
}}
.matrix-wrap table {{
    border-collapse:collapse; font-family:'Courier New',monospace;
    font-size:.82rem; color:#e2e8f0; margin:0 auto;
    background:rgba(10,14,45,.96);
    border:1px solid rgba(99,102,241,.35); border-radius:6px; overflow:hidden;
}}
.matrix-wrap td {{
    padding:.28rem .55rem; text-align:right;
    border:1px solid rgba(99,102,241,.18);
    color:#c7d2fe!important; white-space:nowrap;
}}
.matrix-wrap td.row-label {{
    color:#818cf8!important; font-size:.75rem;
    font-family:'Segoe UI',sans-serif; padding-right:.4rem; text-align:left;
    background:rgba(15,20,55,.95); border-right:1px solid rgba(99,102,241,.35);
}}
.matrix-bracket {{ color:#6366f1; font-size:2rem; vertical-align:middle; user-select:none; }}
.matrix-name {{ color:#a78bfa; font-weight:700; font-size:.9rem; margin-bottom:.3rem; }}

hr {{ border-color:rgba(99,102,241,.20)!important; }}
[data-testid="stCheckbox"] label {{ color:#cbd5e1!important; font-size:.88rem; }}
</style>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ──────────────────────────────────────────────────────────────────
def render_matrix(M: np.ndarray, name: str, cap: int = 6, fmt: str = ".4g") -> str:
    """NumPy matrisini güzel HTML tablosuna çevirir."""
    r, c = M.shape
    Md = M[:cap, :cap]
    rows_html = ""
    for i, row in enumerate(Md):
        cells = "".join(f"<td>{v:{fmt}}</td>" for v in row)
        rows_html += f"<tr>{cells}</tr>"

    trunc_note = ""
    if r > cap or c > cap:
        trunc_note = f'<div style="color:#64748b;font-size:.73rem;margin-top:.2rem;text-align:center;">ilk {cap}×{cap} gösteriliyor ({r}×{c} matris)</div>'

    return f"""
    <div class="matrix-wrap">
        <div class="matrix-name">{name}</div>
        <table>{rows_html}</table>
        {trunc_note}
    </div>"""


def vandermonde(x, d):
    return np.vander(x, N=d+1, increasing=True)

def fit_ekk(A, y):
    AtA = A.T @ A
    kappa = np.linalg.cond(AtA)
    try:    beta = np.linalg.solve(AtA, A.T @ y)
    except: beta = np.full(A.shape[1], np.nan)
    return beta, AtA, kappa

def fit_qr(A, y):
    Q, R = np.linalg.qr(A)
    return np.linalg.solve(R, Q.T @ y), Q, R

def poly_eval(beta, x):
    return sum(beta[i]*x**i for i in range(len(beta)))

def r2_score(y, yh):
    tss = np.sum((y-np.mean(y))**2)
    return 1 - np.sum((y-yh)**2)/tss if tss > 0 else 0.0

def safe_eval(expr, x_arr):
    ns = {"x":x_arr,"np":np,"sin":np.sin,"cos":np.cos,"tan":np.tan,
          "exp":np.exp,"log":np.log,"sqrt":np.sqrt,"abs":np.abs,
          "pi":np.pi,"e":np.e,"sinh":np.sinh,"cosh":np.cosh,"log10":np.log10}
    try:    return np.asarray(eval(compile(expr,"<s>","eval"),{"__builtins__":{}},ns),dtype=float)
    except: return None


# ──────────────────────────────────────────────────────────────────
# VERİ SETLERİ
# ──────────────────────────────────────────────────────────────────
rng = np.random.RandomState(42)
PRESETS = {
    "🏠 Ev Fiyatları":        {"x":[50,65,80,95,110,130,150,170,200,230],"y":[120,150,175,210,240,290,340,380,450,520],"d":2,"desc":"Metrekare → Fiyat (bin ₺)"},
    "📡 Radar Sinyalleri":    {"x":np.linspace(0,4*np.pi,18).tolist(),"y":(np.sin(np.linspace(0,4*np.pi,18))+.3*rng.randn(18)).tolist(),"d":7,"desc":"Zaman → Sinyal gücü"},
    "🌡️ Sıcaklık (Aylık)":   {"x":list(range(1,13)),"y":[3.2,4.8,9.5,15.1,20.4,25.8,28.2,27.6,22.3,15.9,9.1,4.5],"d":5,"desc":"Ay → Ortalama °C"},
    "✏️ Manuel Giriş":        {"x":[1,2,3,4,5,6,7],"y":[2.1,4.5,9.2,16.8,26.1,37.9,51.4],"d":3,"desc":"Kendi verilerinizi girin"},
}


# ──────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Kontrol Paneli")
    st.divider()

    preset_name = st.selectbox("Veri Seti", list(PRESETS.keys()), label_visibility="collapsed")
    p = PRESETS[preset_name]
    st.caption(f"ℹ️ {p['desc']}")

    st.divider()
    degree = st.slider("📐 Polinom Derecesi (d)", 1, 12, p["d"])

    st.divider()
    st.markdown("**Grafik**")
    show_ekk  = st.checkbox("EKK eğrisi",        value=True)
    show_qr   = st.checkbox("QR eğrisi",          value=True)
    show_cust = st.checkbox("Özel denklem",        value=True)
    show_res  = st.checkbox("Kalıntı çizgileri",   value=False)
    show_band = st.checkbox("QR ±σ bölgesi",       value=False)

    st.divider()
    mat_cap = st.slider("Matris görüntü boyutu", 3, 8, 5)

    st.divider()
    st.caption("κ(AᵀA) > 10¹²  →  EKK güvenilmez")


# ──────────────────────────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>∑ EKK vs QR Ayrışımı</h1>
  <p>Karşılaştırmalı Hesaplama Motoru · Matematik Bitirme Projesi</p>
</div>""", unsafe_allow_html=True)
st.divider()


# ──────────────────────────────────────────────────────────────────
# BÖLÜM 1 — VERİ
# ──────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-header">① Veri Seti</div>', unsafe_allow_html=True)

col_data, col_eq = st.columns([1.1, 1], gap="medium")

with col_data:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.caption("Satır ekleyip silebilir, değerleri değiştirebilirsiniz.")
    init_df = pd.DataFrame({"x": list(p["x"]), "y": list(p["y"])})
    edited = st.data_editor(init_df, num_rows="dynamic", use_container_width=True,
        column_config={
            "x": st.column_config.NumberColumn("x (Bağımsız)", format="%.4f"),
            "y": st.column_config.NumberColumn("y (Bağımlı)",  format="%.4f"),
        }, key=f"de_{preset_name}")
    st.markdown("</div>", unsafe_allow_html=True)

with col_eq:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**🧮 Özel Denklem  f(x)**")
    st.caption("NumPy destekli: `sin(x)`, `exp(-x)`, `x**3 - 2*x`, `log(x+1)`")
    custom_expr = st.text_input("f(x) =", value="sin(x)", label_visibility="collapsed", placeholder="örn: x**2 + sin(x)")

    if custom_expr:
        _t = safe_eval(custom_expr, np.array([1.,2.,3.]))
        if _t is not None:
            st.success(f"✅  f(1)={_t[0]:.3f}  ·  f(2)={_t[1]:.3f}  ·  f(3)={_t[2]:.3f}")
        else:
            st.error("❌ Geçersiz ifade.")
            custom_expr = ""

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("Bu eğri grafik üzerinde **sarı** renkte gösterilir. Regresyon eğrileriyle karşılaştırın.")
    st.markdown("</div>", unsafe_allow_html=True)


# ── Veri hazırlama ──
df_c = edited.dropna()
if len(df_c) < degree + 2:
    st.error(f"d={degree} için en az {degree+2} nokta gerekli. Veri ekleyin veya dereceyi azaltın.")
    st.stop()

x = df_c["x"].to_numpy(float)
y = df_c["y"].to_numpy(float)
n = len(x)

A         = vandermonde(x, degree)
b_ekk, AtA, kappa = fit_ekk(A, y)
b_qr, Q, R = fit_qr(A, y)

xf       = np.linspace(x.min(), x.max(), 600)
yf_ekk   = poly_eval(b_ekk, xf)
yf_qr    = poly_eval(b_qr,  xf)
yh_ekk   = poly_eval(b_ekk, x)
yh_qr    = poly_eval(b_qr,  x)
e_ekk    = y - yh_ekk
e_qr     = y - yh_qr
yf_cust  = safe_eval(custom_expr, xf) if custom_expr else None
KAPPA_T  = 1e12


# ──────────────────────────────────────────────────────────────────
# BÖLÜM 2 — KARARLILlIK & METRİKLER
# ──────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-header">② Sayısal Kararlılık</div>', unsafe_allow_html=True)

mc = st.columns(5)
with mc[0]: st.metric("n  (Nokta sayısı)", n)
with mc[1]: st.metric("d  (Derece)", degree)
with mc[2]: st.metric("κ(AᵀA)", f"{kappa:.2e}")
with mc[3]: st.metric("R²  EKK", f"{r2_score(y,yh_ekk):.5f}")
with mc[4]: st.metric("R²  QR",  f"{r2_score(y,yh_qr):.5f}")

if kappa > KAPPA_T:
    st.error(f"🚨 **ILL-CONDITIONED** — κ = {kappa:.2e} (eşik 10¹² aşıldı). EKK güvenilmez, QR zorunludur.")
elif kappa > 1e6:
    st.warning(f"⚡ Orta düzey ill-conditioning: κ = {kappa:.2e}. Kritik uygulamalarda QR tercih edin.")
else:
    st.success(f"✅ İyi koşullanmış sistem: κ = {kappa:.2e}")


# ──────────────────────────────────────────────────────────────────
# BÖLÜM 3 — MATEMATİKSEL YAPILAR
# ──────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-header">③ Matematiksel Yapılar</div>', unsafe_allow_html=True)

tA, tAtA, tQ, tR, tBeta, tTh = st.tabs([
    "📐 A  (Tasarım)",
    "🔴 AᵀA  (Normal)",
    "🟢 Q  (Ortogonal)",
    "🔵 R  (Üst Üçgen)",
    "📈 β  (Katsayılar)",
    "📚 Teori",
])

with tA:
    left, right = st.columns([1,1], gap="medium")
    with left:
        st.latex(r"A_{ij} = x_i^{j-1}, \quad A \in \mathbb{R}^{n \times (d+1)}")
        st.caption(f"Boyut: **{A.shape[0]} × {A.shape[1]}**")
    with right:
        st.markdown(render_matrix(A, f"A  [{A.shape[0]}×{A.shape[1]}]", mat_cap), unsafe_allow_html=True)

with tAtA:
    left, right = st.columns([1,1], gap="medium")
    with left:
        st.latex(r"\hat{\beta}_{\text{EKK}} = (A^T A)^{-1} A^T y")
        st.latex(r"\kappa(A^T A) = [\kappa(A)]^2")
        ks = "⚠️ Yüksek" if kappa > KAPPA_T else "✅ OK"
        st.caption(f"κ(AᵀA) = **{kappa:.3e}**  |  {ks}")
    with right:
        st.markdown(render_matrix(AtA, f"AᵀA  [{AtA.shape[0]}×{AtA.shape[1]}]", mat_cap), unsafe_allow_html=True)

with tQ:
    left, right = st.columns([1,1], gap="medium")
    with left:
        st.latex(r"A = QR, \quad Q^T Q = I_{d+1}")
        err = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]), "fro")
        st.caption(f"Ortonormallik hatası: **{err:.2e}**")
        st.success("✅ Ortogonallik doğrulandı") if err < 1e-10 else st.warning(f"Hata: {err:.2e}")
    with right:
        st.markdown(render_matrix(Q, f"Q  [{Q.shape[0]}×{Q.shape[1]}]", mat_cap), unsafe_allow_html=True)

with tR:
    left, right = st.columns([1,1], gap="medium")
    with left:
        st.latex(r"R_{ij} = 0 \text{ for } i > j \quad \text{(üst üçgen)}")
        st.latex(rf"\kappa(R) \approx {np.linalg.cond(R):.3e}")
        st.caption("κ(R) = κ(A) — kare alma yok, daha kararlı.")
    with right:
        st.markdown(render_matrix(R, f"R  [{R.shape[0]}×{R.shape[1]}]", mat_cap), unsafe_allow_html=True)

with tBeta:
    st.latex(r"\hat{y}(x) = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots + \beta_d x^d")
    bdf = pd.DataFrame({
        "Katsayı":    [f"β_{k}" for k in range(degree+1)],
        "EKK":        [f"{v:.10f}" for v in b_ekk],
        "QR":         [f"{v:.10f}" for v in b_qr],
        "|EKK − QR|": [f"{abs(a-b):.3e}" for a,b in zip(b_ekk, b_qr)],
    })
    st.dataframe(bdf, use_container_width=True, hide_index=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("EKK  R²",  f"{r2_score(y,yh_ekk):.8f}")
    with c2: st.metric("EKK  RSS", f"{float(e_ekk@e_ekk):.4e}")
    with c3: st.metric("QR   R²",  f"{r2_score(y,yh_qr):.8f}")
    with c4: st.metric("QR   RSS", f"{float(e_qr@e_qr):.4e}")

with tTh:
    st.markdown("#### EKK vs QR — Hata Büyüme Karşılaştırması")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Normal Denklemler (EKK)**")
        st.latex(r"(A^T A)\,\hat{\beta} = A^T y")
        st.latex(r"\Delta\beta \;\sim\; \kappa^2 \cdot \varepsilon_{\text{mach}}")
        st.error(f"Bu projede: hata ≈ {kappa**2 * 1e-16:.2e}")
    with c2:
        st.markdown("**QR Ayrışımı**")
        st.latex(r"A = QR \;\Rightarrow\; R\,\hat{\beta} = Q^T y")
        st.latex(r"\Delta\beta \;\sim\; \kappa \cdot \varepsilon_{\text{mach}}")
        st.success(f"Bu projede: hata ≈ {kappa * 1e-16:.2e}")
    st.divider()
    st.latex(r"\kappa(R) = \kappa(A), \quad \kappa(A^T A) = [\kappa(A)]^2 \implies \text{QR her zaman daha kararlıdır.}")


# ──────────────────────────────────────────────────────────────────
# BÖLÜM 4 — GRAFİKLER
# ──────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-header">④ Regresyon Grafiği</div>', unsafe_allow_html=True)

PBASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,14,48,.65)",
    font=dict(color="#e2e8f0",family="Segoe UI",size=12),
    legend=dict(bgcolor="rgba(6,8,24,.88)",bordercolor="rgba(99,102,241,.40)",
                borderwidth=1,font=dict(size=11,color="#e2e8f0")),
    margin=dict(l=50,r=20,t=45,b=35),
)
AX = dict(gridcolor="rgba(99,102,241,.13)",zerolinecolor="rgba(99,102,241,.35)",
          color="#94a3b8",linecolor="rgba(99,102,241,.25)")

fig = make_subplots(rows=2,cols=1,row_heights=[.68,.32],shared_xaxes=True,
                    vertical_spacing=.06,
                    subplot_titles=["Polinom Regresyonu","Kalıntı Analizi"])

# Veri
fig.add_trace(go.Scatter(x=x,y=y,mode="markers",name="Gözlem",
    marker=dict(size=9,color="#f8fafc",symbol="circle",line=dict(width=1.5,color="#6366f1"))), row=1,col=1)

# EKK
if show_ekk:
    fig.add_trace(go.Scatter(x=xf,y=yf_ekk,mode="lines",name=f"EKK (d={degree})",
        line=dict(color="#f87171",width=2.2,dash="dash")), row=1,col=1)

# QR
if show_qr:
    if show_band:
        s = np.std(e_qr)
        fig.add_trace(go.Scatter(
            x=np.concatenate([xf,xf[::-1]]),
            y=np.concatenate([yf_qr+s,(yf_qr-s)[::-1]]),
            fill="toself",fillcolor="rgba(52,211,153,.09)",
            line=dict(color="rgba(0,0,0,0)"),name="QR ±σ"), row=1,col=1)
    fig.add_trace(go.Scatter(x=xf,y=yf_qr,mode="lines",name=f"QR (d={degree})",
        line=dict(color="#34d399",width=2.2)), row=1,col=1)

# Özel denklem
if show_cust and yf_cust is not None:
    fig.add_trace(go.Scatter(x=xf,y=yf_cust,mode="lines",name=f"f(x)={custom_expr}",
        line=dict(color="#fbbf24",width=2,dash="dot")), row=1,col=1)

# Kalıntı çizgileri
if show_res:
    for xi,yi,ei,qi in zip(x,y,yh_ekk,yh_qr):
        if show_ekk:
            fig.add_trace(go.Scatter(x=[xi,xi],y=[yi,ei],mode="lines",showlegend=False,
                line=dict(color="rgba(248,113,113,.40)",width=1)), row=1,col=1)
        if show_qr:
            fig.add_trace(go.Scatter(x=[xi,xi],y=[yi,qi],mode="lines",showlegend=False,
                line=dict(color="rgba(52,211,153,.40)",width=1)), row=1,col=1)

# Kalıntı barları
if show_ekk:
    fig.add_trace(go.Bar(x=x,y=e_ekk,name="EKK Kalıntı",
        marker_color="rgba(248,113,113,.70)"), row=2,col=1)
if show_qr:
    fig.add_trace(go.Bar(x=x,y=e_qr,name="QR Kalıntı",
        marker_color="rgba(52,211,153,.70)"), row=2,col=1)

fig.add_hline(y=0,line=dict(color="rgba(200,210,255,.30)",width=1,dash="dot"),row=2,col=1)
fig.update_xaxes(**AX); fig.update_yaxes(**AX)
fig.update_layout(height=640,**PBASE)
st.plotly_chart(fig,use_container_width=True)


# ── Kararlılık analizi (expander) ──
with st.expander("🔬 Derece → Koşul Sayısı İlişkisi"):
    dr = list(range(1, min(14, n-1)))
    ks_list = []
    for d in dr:
        At = vandermonde(x, d)
        ks_list.append(np.linalg.cond(At.T @ At))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dr,y=ks_list,mode="lines+markers",name="κ(AᵀA)",
        line=dict(color="#a78bfa",width=2),marker=dict(size=7,color="#a78bfa")))
    fig2.add_hline(y=KAPPA_T,line=dict(color="#f87171",width=1.5,dash="dash"),
        annotation_text="Eşik 10¹²",annotation_font_color="#f87171")
    fig2.add_vline(x=degree,line=dict(color="#fbbf24",width=1.5,dash="dot"),
        annotation_text=f"d={degree}",annotation_font_color="#fbbf24")
    fig2.update_xaxes(title_text="Polinom Derecesi",**AX)
    fig2.update_yaxes(title_text="κ(AᵀA) — log",type="log",**AX)
    fig2.update_layout(height=340,**PBASE)
    st.plotly_chart(fig2,use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# BÖLÜM 5 — ÖZET RAPOR
# ──────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-header">⑤ Akademik Özet Rapor</div>', unsafe_allow_html=True)

with st.expander("Sonuçları Göster", expanded=True):
    bd = np.linalg.norm(b_ekk - b_qr)
    st.markdown(f"""
| Metrik | EKK | QR |
|:--|--:|--:|
| R² | `{r2_score(y,yh_ekk):.10f}` | `{r2_score(y,yh_qr):.10f}` |
| RSS | `{float(e_ekk@e_ekk):.6e}` | `{float(e_qr@e_qr):.6e}` |
| Max \|kalıntı\| | `{np.max(np.abs(e_ekk)):.6e}` | `{np.max(np.abs(e_qr)):.6e}` |
| ‖β_EKK − β_QR‖₂ | `{bd:.6e}` | — |
| Koşul sayısı | `{kappa:.6e}` | `{np.linalg.cond(R):.6e}` (R) |
| Kararlılık | {"❌ Güvenilmez" if kappa>KAPPA_T else "⚠️ Dikkat" if kappa>1e6 else "✅ İyi"} | ✅ Kararlı |
""")
    st.latex(r"\kappa(R)=\kappa(A),\quad\kappa(A^TA)=[\kappa(A)]^2 \implies \text{QR her zaman daha kararlıdır.}")

st.divider()
st.caption("🎓 Matematik Bitirme Projesi · EKK vs QR · Streamlit + NumPy + Plotly · ε_mach ≈ 2.22×10⁻¹⁶")
