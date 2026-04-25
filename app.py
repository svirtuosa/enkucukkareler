import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
#  Sayfa Ayarları
# ============================================================

st.set_page_config(
    page_title="EKK vs QR Ayrışımı",
    page_icon="📐",
    layout="wide"
)


# ============================================================
#  CSS ve Arka Plan
# ============================================================

def set_background(image_path: str = "image_13.PNG") -> None:
    path = Path(image_path)

    if path.exists():
        encoded = base64.b64encode(path.read_bytes()).decode()
        background_css = f"""
        .stApp {{
            background:
                linear-gradient(rgba(0,0,0,0.80), rgba(0,0,0,0.80)),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        """
    else:
        background_css = """
        .stApp {
            background: linear-gradient(135deg, #0f172a, #020617);
        }
        """

    st.markdown(
        f"""
        <style>
        {background_css}

        html, body, [class*="css"] {{
            color: #f1f5f9;
        }}

        h1, h2, h3, h4 {{
            color: #ffffff;
        }}

        .stButton > button {{
            background: rgba(255, 255, 255, 0.12);
            color: white;
            border: 1px solid rgba(255,255,255,0.25);
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            transition: 0.3s;
        }}

        .stButton > button:hover {{
            background: rgba(255, 255, 255, 0.25);
            border-color: white;
        }}

        .stSelectbox, .stSlider, .stNumberInput, .stDataFrame {{
            background: rgba(15, 23, 42, 0.55);
            border-radius: 14px;
            padding: 0.5rem;
        }}

        div[data-testid="stMetric"] {{
            background: rgba(15, 23, 42, 0.70);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 16px;
            padding: 1rem;
        }}

        div[data-testid="stTabs"] {{
            background: rgba(15, 23, 42, 0.55);
            border-radius: 16px;
            padding: 1rem;
        }}

        .block-container {{
            padding-top: 2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_background()


# ============================================================
#  Yardımcı Fonksiyonlar
# ============================================================

def create_dataset(name: str) -> pd.DataFrame:
    np.random.seed(42)

    if name == "Ev Fiyatları (Lineer)":
        x = np.linspace(50, 250, 12)
        y = 1200 * x + 50_000 + np.random.normal(0, 18_000, len(x))

    elif name == "Radar Sinyalleri (Dalgalı)":
        x = np.linspace(0, 10, 18)
        y = 3 * np.sin(1.7 * x) + 0.45 * x + np.random.normal(0, 0.35, len(x))

    elif name == "Sıcaklık Değişimi":
        x = np.linspace(1, 24, 16)
        y = 18 + 7 * np.sin((np.pi / 12) * (x - 6)) + np.random.normal(0, 0.8, len(x))

    else:
        x = np.arange(1, 8)
        y = np.array([2.1, 2.9, 4.8, 7.2, 11.1, 15.3, 20.2])

    return pd.DataFrame({"x": x, "y": y})


def design_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    return np.vander(x, N=degree + 1, increasing=True)


def normal_equation_solution(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    ATA = A.T @ A
    ATy = A.T @ y
    return np.linalg.solve(ATA, ATy)


def qr_solution(A: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q, R = np.linalg.qr(A, mode="reduced")
    beta = np.linalg.solve(R, Q.T @ y)
    return beta, Q, R


def polynomial_values(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    A_eval = design_matrix(x, len(beta) - 1)
    return A_eval @ beta


def matrix_to_latex(M: np.ndarray, decimals: int = 4, max_rows: int = 10, max_cols: int = 8) -> str:
    M = np.asarray(M)
    rows, cols = M.shape

    shown = M[:max_rows, :max_cols]
    body = []

    for row in shown:
        body.append(" & ".join([f"{value:.{decimals}g}" for value in row]))

    if rows > max_rows:
        body.append(r"\vdots" + " & " * (shown.shape[1] - 1))

    latex = r"\begin{bmatrix}" + r"\\".join(body) + r"\end{bmatrix}"

    if rows > max_rows or cols > max_cols:
        latex += r"\quad \text{(kısaltılmış gösterim)}"

    return latex


def coefficients_to_latex(beta: np.ndarray) -> str:
    terms = []
    for i, b in enumerate(beta):
        if i == 0:
            terms.append(f"{b:.5g}")
        elif i == 1:
            terms.append(f"{b:.5g}x")
        else:
            terms.append(f"{b:.5g}x^{i}")
    return " + ".join(terms)


# ============================================================
#  Başlık
# ============================================================

st.title("📐 Klasik EKK ve QR Ayrışımı Karşılaştırmalı Hesaplama Motoru")

st.markdown(
    """
Bu uygulama, yüksek dereceli polinom yaklaşımlarında klasik normal denklem tabanlı 
**En Küçük Kareler (EKK)** yönteminin sayısal kararsızlığını ve **QR Ayrışımı** yönteminin 
neden daha güvenilir olduğunu deneysel olarak gösterir.
"""
)


# ============================================================
#  Yan Panel
# ============================================================

with st.sidebar:
    st.header("⚙️ Parametreler")

    selected_dataset = st.selectbox(
        "Hazır veri seti seç",
        [
            "Ev Fiyatları (Lineer)",
            "Radar Sinyalleri (Dalgalı)",
            "Sıcaklık Değişimi",
            "Özel Başlangıç Verisi"
        ]
    )

    if "active_dataset" not in st.session_state:
        st.session_state.active_dataset = selected_dataset
        st.session_state.data = create_dataset(selected_dataset)

    if selected_dataset != st.session_state.active_dataset:
        st.session_state.active_dataset = selected_dataset
        st.session_state.data = create_dataset(selected_dataset)

    degree = st.slider(
        "Polinom derecesi",
        min_value=1,
        max_value=12,
        value=3
    )

    show_ekk = st.checkbox("EKK eğrisini göster", value=True)
    show_qr = st.checkbox("QR eğrisini göster", value=True)
    show_residuals = st.checkbox("Kalıntı çizgilerini göster", value=False)

    st.warning(
        "Yüksek derece seçildiğinde Vandermonde matrisi kötü koşullu hale gelebilir."
    )


# ============================================================
#  Veri Girişi
# ============================================================

st.subheader("1. Veri Giriş Katmanı")

edited_data = st.data_editor(
    st.session_state.data,
    num_rows="dynamic",
    use_container_width=True,
    key="data_editor"
)

data = edited_data.dropna()

if len(data) < degree + 1:
    st.error("Seçilen polinom derecesi için yeterli veri noktası yok.")
    st.stop()

x = data["x"].to_numpy(dtype=float)
y = data["y"].to_numpy(dtype=float)

sort_idx = np.argsort(x)
x = x[sort_idx]
y = y[sort_idx]


# ============================================================
#  Matematiksel Hesaplama Motoru
# ============================================================

A = design_matrix(x, degree)
ATA = A.T @ A

condition_number = np.linalg.cond(ATA)

try:
    beta_ekk = normal_equation_solution(A, y)
except np.linalg.LinAlgError:
    beta_ekk = np.full(degree + 1, np.nan)

try:
    beta_qr, Q, R = qr_solution(A, y)
except np.linalg.LinAlgError:
    beta_qr = np.full(degree + 1, np.nan)
    Q = np.full_like(A, np.nan)
    R = np.full((degree + 1, degree + 1), np.nan)

y_hat_ekk = polynomial_values(beta_ekk, x) if np.all(np.isfinite(beta_ekk)) else np.full_like(y, np.nan)
y_hat_qr = polynomial_values(beta_qr, x) if np.all(np.isfinite(beta_qr)) else np.full_like(y, np.nan)

rss_ekk = np.sum((y - y_hat_ekk) ** 2) if np.all(np.isfinite(y_hat_ekk)) else np.nan
rss_qr = np.sum((y - y_hat_qr) ** 2) if np.all(np.isfinite(y_hat_qr)) else np.nan


# ============================================================
#  Sayısal Kararlılık
# ============================================================

st.subheader("2. Sayısal Kararlılık Göstergesi")

col1, col2, col3 = st.columns(3)

col1.metric(
    label=r"Condition Number: $A^T A$",
    value=f"{condition_number:.3e}"
)

col2.metric(
    label="EKK RSS",
    value=f"{rss_ekk:.5g}"
)

col3.metric(
    label="QR RSS",
    value=f"{rss_qr:.5g}"
)

if condition_number > 1e12:
    st.error(
        r"""
        Kritik uyarı: $cond(A^T A) > 10^{12}$ olduğu için klasik EKK çözümü 
        sayısal olarak güvenilmez hale gelmiştir. Bu durumda yuvarlama hataları 
        büyüyebilir ve katsayılar kararsızlaşabilir.
        """
    )
else:
    st.success(
        "Koşul sayısı kritik eşiğin altında. Ancak derece arttıkça sayısal kararsızlık izlenmelidir."
    )


# ============================================================
#  Grafik
# ============================================================

st.subheader("3. EKK ve QR Eğrilerinin Karşılaştırılması")

x_grid = np.linspace(np.min(x), np.max(x), 500)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="Veri Noktaları",
        marker=dict(size=9)
    )
)

if show_ekk and np.all(np.isfinite(beta_ekk)):
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=polynomial_values(beta_ekk, x_grid),
            mode="lines",
            name="Klasik EKK"
        )
    )

if show_qr and np.all(np.isfinite(beta_qr)):
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=polynomial_values(beta_qr, x_grid),
            mode="lines",
            name="QR Ayrışımı"
        )
    )

if show_residuals and np.all(np.isfinite(y_hat_qr)):
    for xi, yi, yqi in zip(x, y, y_hat_qr):
        fig.add_trace(
            go.Scatter(
                x=[xi, xi],
                y=[yi, yqi],
                mode="lines",
                line=dict(dash="dot"),
                showlegend=False,
                hoverinfo="skip"
            )
        )

fig.update_layout(
    template="plotly_dark",
    height=620,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.65)",
    title="Polinom Yaklaşımı ve Kalıntılar",
    xaxis_title="x",
    yaxis_title="y",
    legend_title="Yöntem"
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
#  Matematiksel Mutfak
# ============================================================

st.subheader("4. Matematiksel Mutfak: Matrisler ve LaTeX Açıklamalar")

tab_A, tab_ATA, tab_Q, tab_R, tab_coef = st.tabs(
    ["Tasarım Matrisi A", "Normal Denklem AᵀA", "Ortogonal Q", "Üst Üçgen R", "Katsayılar"]
)

with tab_A:
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

with tab_ATA:
    st.latex(
        r"""
        \hat{\beta}_{EKK}
        =
        (A^T A)^{-1} A^T y
        """
    )
    st.latex(
        r"""
        A^T A
        """
    )
    st.latex(matrix_to_latex(ATA))
    st.markdown(
        """
Normal denklem yöntemi teorik olarak doğru olsa da, pratikte 
$A^T A$ matrisi koşul sayısını yaklaşık olarak karesel biçimde büyütür. 
Bu nedenle yüksek dereceli polinomlarda EKK çözümü sayısal olarak kırılganlaşır.
"""
    )

with tab_Q:
    st.latex(
        r"""
        A = QR, \qquad Q^TQ = I
        """
    )
    st.latex("Q = " + matrix_to_latex(Q))
    st.markdown(
        """
QR ayrışımı, normal denklemdeki $A^T A$ matrisini açıkça oluşturmadan çözüm üretir. 
Bu nedenle yuvarlama hatalarına karşı daha kararlıdır.
"""
    )

with tab_R:
    st.latex(
        r"""
        R\hat{\beta}_{QR} = Q^T y
        """
    )
    st.latex("R = " + matrix_to_latex(R))

with tab_coef:
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Klasik EKK Polinomu")
        if np.all(np.isfinite(beta_ekk)):
            st.latex(r"p_{EKK}(x) = " + coefficients_to_latex(beta_ekk))
            st.dataframe(
                pd.DataFrame(
                    {
                        "Derece": np.arange(len(beta_ekk)),
                        "Katsayı": beta_ekk
                    }
                ),
                use_container_width=True
            )
        else:
            st.error("EKK katsayıları hesaplanamadı.")

    with col_right:
        st.markdown("### QR Polinomu")
        if np.all(np.isfinite(beta_qr)):
            st.latex(r"p_{QR}(x) = " + coefficients_to_latex(beta_qr))
            st.dataframe(
                pd.DataFrame(
                    {
                        "Derece": np.arange(len(beta_qr)),
                        "Katsayı": beta_qr
                    }
                ),
                use_container_width=True
            )
        else:
            st.error("QR katsayıları hesaplanamadı.")


# ============================================================
#  Akademik Sonuç
# ============================================================

st.subheader("5. Akademik Yorum")

st.markdown(
    f"""
Seçilen polinom derecesi **{degree}** için hesaplanan koşul sayısı:

$$
cond(A^T A) = {condition_number:.3e}
$$

Bu değer büyüdükçe klasik EKK yönteminde kullanılan normal denklem yaklaşımı 
sayısal olarak hassaslaşır. QR ayrışımı ise problemi ortogonal dönüşümler üzerinden 
çözdüğü için özellikle yüksek dereceli polinom regresyonlarında daha kararlı sonuçlar üretir.
"""
)
