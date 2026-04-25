import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="EKK vs QR Ayrışımı | Sayısal Kararlılık Motoru",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# BACKGROUND & THEME
# ============================================================

def load_background(image_path: str = "image_13.PNG") -> str:
    path = Path(image_path)

    if path.exists():
        encoded = base64.b64encode(path.read_bytes()).decode()
        return f"""
        .stApp {{
            background:
                linear-gradient(rgba(0,0,0,0.80), rgba(0,0,0,0.80)),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        """

    return """
    .stApp {
        background: radial-gradient(circle at top, #1e293b 0%, #020617 60%);
    }
    """


def apply_custom_css() -> None:
    st.markdown(
        f"""
        <style>
        {load_background()}

        html, body, [class*="css"] {{
            color: #f8fafc;
        }}

        h1, h2, h3, h4 {{
            color: #ffffff;
            font-weight: 800;
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}

        .glass-card {{
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 20px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 10px 35px rgba(0,0,0,0.35);
        }}

        div[data-testid="stMetric"] {{
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 18px;
            padding: 1rem;
        }}

        .stButton > button {{
            background: rgba(255, 255, 255, 0.12);
            color: white;
            border: 1px solid rgba(255,255,255,0.25);
            border-radius: 14px;
            padding: 0.65rem 1.1rem;
            transition: 0.25s ease-in-out;
        }}

        .stButton > button:hover {{
            background: rgba(255, 255, 255, 0.25);
            border-color: white;
            transform: translateY(-1px);
        }}

        .stSelectbox, .stSlider, .stNumberInput, .stCheckbox, .stDataFrame {{
            background: rgba(15, 23, 42, 0.48);
            border-radius: 16px;
            padding: 0.45rem;
        }}

        div[data-testid="stTabs"] {{
            background: rgba(15, 23, 42, 0.58);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 1rem;
        }}

        .small-text {{
            color: #cbd5e1;
            font-size: 0.95rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


apply_custom_css()


# ============================================================
# DATASETS
# ============================================================

def make_dataset(dataset_name: str, noise_scale: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    if dataset_name == "Ev Fiyatları (Lineer)":
        x = np.linspace(50, 250, 18)
        y = 1250 * x + 55_000 + rng.normal(0, noise_scale * 20_000, size=len(x))

    elif dataset_name == "Radar Sinyalleri (Dalgalı)":
        x = np.linspace(0, 10, 24)
        y = (
            3.2 * np.sin(1.65 * x)
            + 1.1 * np.cos(0.7 * x)
            + 0.38 * x
            + rng.normal(0, noise_scale * 0.45, size=len(x))
        )

    elif dataset_name == "Sıcaklık Değişimi":
        x = np.linspace(1, 24, 24)
        y = (
            19
            + 7.5 * np.sin((np.pi / 12) * (x - 6))
            + 1.2 * np.cos((np.pi / 6) * x)
            + rng.normal(0, noise_scale * 0.85, size=len(x))
        )

    elif dataset_name == "Finansal Trend (Üstelimsi)":
        x = np.linspace(0, 12, 22)
        y = 20 + 4.2 * x + 0.55 * x**2 + rng.normal(0, noise_scale * 5.0, size=len(x))

    elif dataset_name == "Deneysel Ölçüm (Kübik)":
        x = np.linspace(-4, 4, 21)
        y = 2.5 - 1.2 * x + 0.8 * x**2 - 0.25 * x**3 + rng.normal(
            0, noise_scale * 2.0, size=len(x)
        )

    elif dataset_name == "Yüksek Derece Testi":
        x = np.linspace(-1, 1, 28)
        y = 1 / (1 + 25 * x**2) + rng.normal(0, noise_scale * 0.015, size=len(x))

    else:
        x = np.arange(1, 9)
        y = np.array([2.1, 3.2, 4.7, 7.9, 11.4, 15.6, 20.5, 27.1])

    return pd.DataFrame({"x": x, "y": y})


# ============================================================
# LINEAR ALGEBRA ENGINE
# ============================================================

def design_matrix(x_values: np.ndarray, polynomial_degree: int) -> np.ndarray:
    return np.vander(x_values, N=polynomial_degree + 1, increasing=True)


def solve_by_normal_equations(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.solve(A.T @ A, A.T @ y)


def solve_by_qr(A: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q, R = np.linalg.qr(A, mode="reduced")
    beta = np.linalg.solve(R, Q.T @ y)
    return beta, Q, R


def evaluate_polynomial(beta: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    A_eval = design_matrix(x_values, len(beta) - 1)
    return A_eval @ beta


def safe_relative_difference(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(b)
    if denominator == 0:
        return np.nan
    return np.linalg.norm(a - b) / denominator


def compute_statistics(y: np.ndarray, y_hat: np.ndarray, number_of_parameters: int) -> dict:
    residuals = y - y_hat
    rss = float(np.sum(residuals**2))
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))

    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - rss / tss if tss != 0 else np.nan

    n = len(y)
    adjusted_r2 = (
        1 - (1 - r2) * (n - 1) / (n - number_of_parameters)
        if n > number_of_parameters and np.isfinite(r2)
        else np.nan
    )

    return {
        "RSS": rss,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Adjusted R2": adjusted_r2,
    }


def matrix_to_latex(
    matrix: np.ndarray,
    decimals: int = 4,
    max_rows: int = 10,
    max_cols: int = 8
) -> str:
    matrix = np.asarray(matrix)
    rows, cols = matrix.shape
    shown = matrix[:max_rows, :max_cols]

    latex_rows = []
    for row in shown:
        latex_rows.append(" & ".join(f"{value:.{decimals}g}" for value in row))

    if rows > max_rows:
        latex_rows.append(r"\vdots" + (" & " * (shown.shape[1] - 1)))

    latex = r"\begin{bmatrix}" + r"\\".join(latex_rows) + r"\end{bmatrix}"

    if rows > max_rows or cols > max_cols:
        latex += r"\quad \text{(kısaltılmış)}"

    return latex


def polynomial_to_latex(beta: np.ndarray) -> str:
    terms = []

    for degree, coefficient in enumerate(beta):
        sign = "+" if coefficient >= 0 else "-"
        absolute = abs(coefficient)

        if degree == 0:
            term = f"{absolute:.5g}"
        elif degree == 1:
            term = f"{absolute:.5g}x"
        else:
            term = f"{absolute:.5g}x^{degree}"

        if degree == 0:
            terms.append(term if coefficient >= 0 else f"-{term}")
        else:
            terms.append(f" {sign} {term}")

    return "".join(terms)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Kontrol Paneli")

    dataset_name = st.selectbox(
        "Hazır veri seti",
        [
            "Ev Fiyatları (Lineer)",
            "Radar Sinyalleri (Dalgalı)",
            "Sıcaklık Değişimi",
            "Finansal Trend (Üstelimsi)",
            "Deneysel Ölçüm (Kübik)",
            "Yüksek Derece Testi",
            "Özel Başlangıç Verisi",
        ]
    )

    polynomial_degree = st.slider(
        "Polinom derecesi",
        min_value=1,
        max_value=15,
        value=5
    )

    noise_scale = st.slider(
        "Veri gürültüsü",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1
    )

    random_seed = st.number_input(
        "Rastgelelik tohumu",
        min_value=1,
        max_value=9999,
        value=42,
        step=1
    )

    st.divider()

    show_ekk_curve = st.checkbox("EKK eğrisini göster", value=True)
    show_qr_curve = st.checkbox("QR eğrisini göster", value=True)
    show_residual_lines = st.checkbox("Kalıntı çizgilerini göster", value=True)
    show_residual_bar = st.checkbox("Kalıntı bar grafiğini göster", value=True)
    show_condition_comparison = st.checkbox("Koşul sayısı karşılaştırmasını göster", value=True)

    st.divider()

    scale_x = st.checkbox(
        "x değerlerini standartlaştır",
        value=False,
        help="Yüksek dereceli polinomlarda sayısal kararlılığı iyileştirebilir."
    )


if (
    "dataset_name" not in st.session_state
    or st.session_state.dataset_name != dataset_name
    or st.session_state.noise_scale != noise_scale
    or st.session_state.random_seed != random_seed
):
    st.session_state.dataset_name = dataset_name
    st.session_state.noise_scale = noise_scale
    st.session_state.random_seed = random_seed
    st.session_state.data = make_dataset(dataset_name, noise_scale, random_seed)


# ============================================================
# HEADER
# ============================================================

st.title("📐 EKK ve QR Ayrışımı Karşılaştırmalı Hesaplama Motoru")

st.markdown(
    """
<div class="glass-card">
Bu uygulama, klasik normal denklem tabanlı <b>En Küçük Kareler</b> yöntemi ile 
<b>QR Ayrışımı</b> yöntemini karşılaştırır. Amaç, özellikle yüksek dereceli polinom 
regresyonunda <b>ill-conditioned</b> yapıların EKK çözümünü nasıl bozduğunu ve QR yönteminin 
neden daha kararlı olduğunu deneysel ve akademik biçimde göstermektir.
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# DATA EDITOR
# ============================================================

st.subheader("1. Veri Giriş Katmanı")

edited_data = st.data_editor(
    st.session_state.data,
    num_rows="dynamic",
    use_container_width=True,
    key="data_editor"
)

data = edited_data.dropna().copy()

if "x" not in data.columns or "y" not in data.columns:
    st.error("Tabloda mutlaka `x` ve `y` sütunları bulunmalıdır.")
    st.stop()

data["x"] = pd.to_numeric(data["x"], errors="coerce")
data["y"] = pd.to_numeric(data["y"], errors="coerce")
data = data.dropna()

if len(data) < polynomial_degree + 1:
    st.error("Seçilen polinom derecesi için yeterli veri noktası yok.")
    st.stop()

data = data.sort_values("x")
x_original = data["x"].to_numpy(dtype=float)
y = data["y"].to_numpy(dtype=float)

if scale_x:
    x_mean = np.mean(x_original)
    x_std = np.std(x_original)

    if x_std == 0:
        st.error("x değerlerinin standart sapması sıfır olduğu için standartlaştırma yapılamaz.")
        st.stop()

    x = (x_original - x_mean) / x_std
else:
    x = x_original.copy()


# ============================================================
# COMPUTATION
# ============================================================

A = design_matrix(x, polynomial_degree)
ATA = A.T @ A

condition_A = np.linalg.cond(A)
condition_ATA = np.linalg.cond(ATA)

ekk_failed = False
qr_failed = False

try:
    beta_ekk = solve_by_normal_equations(A, y)
except np.linalg.LinAlgError:
    beta_ekk = np.full(polynomial_degree + 1, np.nan)
    ekk_failed = True

try:
    beta_qr, Q, R = solve_by_qr(A, y)
except np.linalg.LinAlgError:
    beta_qr = np.full(polynomial_degree + 1, np.nan)
    Q = np.full_like(A, np.nan)
    R = np.full((polynomial_degree + 1, polynomial_degree + 1), np.nan)
    qr_failed = True

y_hat_ekk = (
    evaluate_polynomial(beta_ekk, x)
    if np.all(np.isfinite(beta_ekk))
    else np.full_like(y, np.nan)
)

y_hat_qr = (
    evaluate_polynomial(beta_qr, x)
    if np.all(np.isfinite(beta_qr))
    else np.full_like(y, np.nan)
)

stats_ekk = (
    compute_statistics(y, y_hat_ekk, polynomial_degree + 1)
    if np.all(np.isfinite(y_hat_ekk))
    else None
)

stats_qr = (
    compute_statistics(y, y_hat_qr, polynomial_degree + 1)
    if np.all(np.isfinite(y_hat_qr))
    else None
)

coefficient_difference = (
    safe_relative_difference(beta_ekk, beta_qr)
    if np.all(np.isfinite(beta_ekk)) and np.all(np.isfinite(beta_qr))
    else np.nan
)


# ============================================================
# NUMERICAL STABILITY DASHBOARD
# ============================================================

st.subheader("2. Sayısal Kararlılık Paneli")

m1, m2, m3, m4 = st.columns(4)

m1.metric("cond(A)", f"{condition_A:.3e}")
m2.metric("cond(AᵀA)", f"{condition_ATA:.3e}")
m3.metric("Katsayı Farkı", f"{coefficient_difference:.3e}")
m4.metric("Polinom Derecesi", polynomial_degree)

if condition_ATA > 1e12:
    st.error(
        r"""
Kritik uyarı: $cond(A^T A) > 10^{12}$ olduğu için normal denklem tabanlı EKK çözümü 
sayısal olarak güvenilmez hale gelmiştir. Bu durumda küçük yuvarlama hataları katsayılarda 
büyük sapmalara neden olabilir.
"""
    )
elif condition_ATA > 1e8:
    st.warning(
        r"""
Dikkat: $cond(A^T A)$ oldukça yüksek. EKK çözümü henüz tamamen çökmediği halde 
katsayılar sayısal hatalara karşı hassas olabilir.
"""
    )
else:
    st.success("Koşul sayısı kritik seviyenin altında görünüyor.")


# ============================================================
# MAIN PLOT
# ============================================================

st.subheader("3. EKK ve QR Eğrilerinin Görsel Karşılaştırması")

x_grid_original = np.linspace(np.min(x_original), np.max(x_original), 600)

if scale_x:
    x_grid = (x_grid_original - np.mean(x_original)) / np.std(x_original)
else:
    x_grid = x_grid_original.copy()

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=x_original,
        y=y,
        mode="markers",
        name="Gözlem Noktaları",
        marker=dict(size=9)
    )
)

if show_ekk_curve and np.all(np.isfinite(beta_ekk)):
    fig.add_trace(
        go.Scatter(
            x=x_grid_original,
            y=evaluate_polynomial(beta_ekk, x_grid),
            mode="lines",
            name="Klasik EKK",
            line=dict(width=3)
        )
    )

if show_qr_curve and np.all(np.isfinite(beta_qr)):
    fig.add_trace(
        go.Scatter(
            x=x_grid_original,
            y=evaluate_polynomial(beta_qr, x_grid),
            mode="lines",
            name="QR Ayrışımı",
            line=dict(width=3, dash="dash")
        )
    )

if show_residual_lines and np.all(np.isfinite(y_hat_qr)):
    for xi, yi, yqi in zip(x_original, y, y_hat_qr):
        fig.add_trace(
            go.Scatter(
                x=[xi, xi],
                y=[yi, yqi],
                mode="lines",
                name="QR Kalıntısı",
                line=dict(width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip"
            )
        )

fig.update_layout(
    template="plotly_dark",
    height=620,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.72)",
    title="Polinom Regresyonu: Normal Denklem EKK vs QR Ayrışımı",
    xaxis_title="x",
    yaxis_title="y",
    legend_title="Gösterimler"
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# RESIDUAL ANALYSIS
# ============================================================

if show_residual_bar:
    st.subheader("4. Kalıntı Analizi")

    residual_ekk = y - y_hat_ekk
    residual_qr = y - y_hat_qr

    residual_fig = go.Figure()

    residual_fig.add_trace(
        go.Bar(
            x=x_original,
            y=residual_ekk,
            name="EKK Kalıntıları"
        )
    )

    residual_fig.add_trace(
        go.Bar(
            x=x_original,
            y=residual_qr,
            name="QR Kalıntıları"
        )
    )

    residual_fig.update_layout(
        template="plotly_dark",
        barmode="group",
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.72)",
        title="Kalıntıların Karşılaştırılması",
        xaxis_title="x",
        yaxis_title="Kalıntı"
    )

    st.plotly_chart(residual_fig, use_container_width=True)


# ============================================================
# CONDITION COMPARISON BY DEGREE
# ============================================================

if show_condition_comparison:
    st.subheader("5. Dereceye Göre Koşul Sayısı Analizi")

    degrees = np.arange(1, min(16, len(x)) + 1)
    cond_A_values = []
    cond_ATA_values = []

    for d in degrees:
        A_d = design_matrix(x, d)
        cond_A_values.append(np.linalg.cond(A_d))
        cond_ATA_values.append(np.linalg.cond(A_d.T @ A_d))

    cond_fig = go.Figure()

    cond_fig.add_trace(
        go.Scatter(
            x=degrees,
            y=cond_A_values,
            mode="lines+markers",
            name="cond(A)"
        )
    )

    cond_fig.add_trace(
        go.Scatter(
            x=degrees,
            y=cond_ATA_values,
            mode="lines+markers",
            name="cond(AᵀA)"
        )
    )

    cond_fig.add_hline(
        y=1e12,
        line_dash="dash",
        annotation_text="Kritik eşik: 10¹²"
    )

    cond_fig.update_layout(
        template="plotly_dark",
        height=460,
        yaxis_type="log",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.72)",
        title="Polinom Derecesi Arttıkça Koşul Sayısının Büyümesi",
        xaxis_title="Polinom Derecesi",
        yaxis_title="Koşul Sayısı - Log Ölçek"
    )

    st.plotly_chart(cond_fig, use_container_width=True)


# ============================================================
# STATISTICS TABLE
# ============================================================

st.subheader("6. Hata Metrikleri")

metrics_table = pd.DataFrame(
    {
        "Metrik": ["RSS", "RMSE", "MAE", "R²", "Düzeltilmiş R²"],
        "Klasik EKK": [
            stats_ekk["RSS"] if stats_ekk else np.nan,
            stats_ekk["RMSE"] if stats_ekk else np.nan,
            stats_ekk["MAE"] if stats_ekk else np.nan,
            stats_ekk["R2"] if stats_ekk else np.nan,
            stats_ekk["Adjusted R2"] if stats_ekk else np.nan,
        ],
        "QR Ayrışımı": [
            stats_qr["RSS"] if stats_qr else np.nan,
            stats_qr["RMSE"] if stats_qr else np.nan,
            stats_qr["MAE"] if stats_qr else np.nan,
            stats_qr["R2"] if stats_qr else np.nan,
            stats_qr["Adjusted R2"] if stats_qr else np.nan,
        ],
    }
)

st.dataframe(metrics_table, use_container_width=True)


# ============================================================
# MATHEMATICAL KITCHEN
# ============================================================

st.subheader("7. Matematiksel Mutfak: LaTeX ve Matrisler")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Tasarım Matrisi A",
        "Normal Denklem",
        "QR Ayrışımı",
        "Q Matrisi",
        "R Matrisi",
        "Polinomlar",
    ]
)

with tab1:
    st.latex(
        r"""
        A =
        \begin{bmatrix}
        1 & x_1 & x_1^2 & \cdots & x_1^n \\
        1 & x_2 & x_2^2 & \cdots & x_2^n \\
        \vdots & \vdots & \vdots & \ddots & \vdots \\
        1 & x_m & x_m^2 & \cdots & x_m^n
        \end{bmatrix}
        """
    )
    st.latex("A = " + matrix_to_latex(A))

with tab2:
    st.latex(
        r"""
        \hat{\beta}_{EKK}
        =
        (A^T A)^{-1} A^T y
        """
    )
    st.latex(r"A^T A = " + matrix_to_latex(ATA))
    st.markdown(
        """
Normal denklem yaklaşımı teorik olarak geçerlidir; fakat pratikte 
$A^TA$ matrisinin kurulması, koşul sayısını büyüterek problemi sayısal olarak hassaslaştırır.
"""
    )

with tab3:
    st.latex(
        r"""
        A = QR, \qquad Q^TQ = I
        """
    )
    st.latex(
        r"""
        A\beta \approx y
        \quad \Longrightarrow \quad
        QR\beta \approx y
        """
    )
    st.latex(
        r"""
        R\hat{\beta}_{QR} = Q^Ty
        """
    )
    st.markdown(
        """
QR yöntemi $A^TA$ matrisini doğrudan oluşturmaz. Bu nedenle özellikle Vandermonde tipi 
tasarım matrislerinde normal denkleme göre daha kararlı bir çözüm üretir.
"""
    )

with tab4:
    st.latex(r"Q = " + matrix_to_latex(Q))
    st.latex(r"Q^TQ \approx I")

with tab5:
    st.latex(r"R = " + matrix_to_latex(R))
    st.markdown("R matrisi üst üçgen yapıdadır ve çözüm geri yerine koyma mantığıyla elde edilir.")

with tab6:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Klasik EKK")
        if not ekk_failed:
            st.latex(r"p_{EKK}(x) = " + polynomial_to_latex(beta_ekk))
            st.dataframe(
                pd.DataFrame(
                    {
                        "Derece": np.arange(len(beta_ekk)),
                        "Katsayı": beta_ekk,
                    }
                ),
                use_container_width=True
            )
        else:
            st.error("EKK katsayıları hesaplanamadı.")

    with c2:
        st.markdown("### QR Ayrışımı")
        if not qr_failed:
            st.latex(r"p_{QR}(x) = " + polynomial_to_latex(beta_qr))
            st.dataframe(
                pd.DataFrame(
                    {
                        "Derece": np.arange(len(beta_qr)),
                        "Katsayı": beta_qr,
                    }
                ),
                use_container_width=True
            )
        else:
            st.error("QR katsayıları hesaplanamadı.")


# ============================================================
# ACADEMIC INTERPRETATION
# ============================================================

st.subheader("8. Akademik Sonuç Yorumu")

st.markdown(
    f"""
<div class="glass-card">

Seçilen polinom derecesi <b>{polynomial_degree}</b> için hesaplanan temel kararlılık değerleri:

<br><br>

<b>cond(A)</b> = {condition_A:.3e}  
<br>
<b>cond(AᵀA)</b> = {condition_ATA:.3e}  
<br>
<b>EKK ve QR katsayıları arasındaki göreli fark</b> = {coefficient_difference:.3e}

<br><br>

Normal denklem yönteminde çözüm:

<br><br>

$$
\\hat{{\\beta}} = (A^TA)^{{-1}}A^Ty
$$

<br>

formülüyle elde edilir. Ancak bu yaklaşımda $A^TA$ matrisi kurulduğu için sayısal kararlılık 
zayıflayabilir. Özellikle yüksek dereceli polinomlarda Vandermonde matrisi kötü koşullu hale gelir.

<br><br>

QR ayrışımı ise:

<br><br>

$$
A = QR
$$

<br>

temeline dayanır ve $A^TA$ matrisini doğrudan oluşturmadan çözüm yaptığı için daha güvenilir 
bir hesaplama yolu sunar.

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# DOWNLOAD SECTION
# ============================================================

st.subheader("9. Sonuçları Dışa Aktar")

result_df = pd.DataFrame(
    {
        "x": x_original,
        "y": y,
        "EKK_Tahmin": y_hat_ekk,
        "QR_Tahmin": y_hat_qr,
        "EKK_Kalıntı": y - y_hat_ekk,
        "QR_Kalıntı": y - y_hat_qr,
    }
)

csv = result_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Sonuçları CSV olarak indir",
    data=csv,
    file_name="ekk_qr_sonuclari.csv",
    mime="text/csv"
)
