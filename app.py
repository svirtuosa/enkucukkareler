import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="EKK & QR Karşılaştırma Laboratuvarı",
    page_icon="📐",
    layout="wide"
)


# ============================================================
# TASARIM
# ============================================================

def set_background(image_path="image_13.PNG"):
    path = Path(image_path)

    if path.exists():
        encoded = base64.b64encode(path.read_bytes()).decode()
        bg = f"""
        .stApp {{
            background:
                linear-gradient(rgba(0,0,0,0.82), rgba(0,0,0,0.82)),
                url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        """
    else:
        bg = """
        .stApp {
            background: radial-gradient(circle at top, #1e293b, #020617 65%);
        }
        """

    st.markdown(
        f"""
        <style>
        {bg}

        html, body, [class*="css"] {{
            color: #f8fafc;
        }}

        h1, h2, h3 {{
            color: white;
            font-weight: 800;
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}

        .glass {{
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 22px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 12px 35px rgba(0,0,0,0.35);
        }}

        div[data-testid="stMetric"] {{
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 18px;
            padding: 1rem;
        }}

        .stButton > button {{
            background: rgba(255,255,255,0.12);
            color: white;
            border: 1px solid rgba(255,255,255,0.25);
            border-radius: 14px;
        }}

        .stButton > button:hover {{
            background: rgba(255,255,255,0.25);
            border-color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_background()


# ============================================================
# MATEMATİK MOTORU
# ============================================================

def design_matrix(x, degree):
    return np.vander(x, N=degree + 1, increasing=True)


def solve_normal_equation(A, y):
    return np.linalg.solve(A.T @ A, A.T @ y)


def solve_qr(A, y):
    Q, R = np.linalg.qr(A, mode="reduced")
    beta = np.linalg.solve(R, Q.T @ y)
    return beta, Q, R


def predict(beta, x):
    return design_matrix(x, len(beta) - 1) @ beta


def stats(y, yhat):
    residual = y - yhat
    rss = np.sum(residual ** 2)
    rmse = np.sqrt(np.mean(residual ** 2))
    mae = np.mean(np.abs(residual))
    tss = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - rss / tss if tss != 0 else np.nan
    return rss, rmse, mae, r2


def matrix_latex(M, max_rows=8, max_cols=7):
    M = np.asarray(M)
    shown = M[:max_rows, :max_cols]

    rows = []
    for row in shown:
        rows.append(" & ".join(f"{v:.4g}" for v in row))

    if M.shape[0] > max_rows:
        rows.append(r"\vdots")

    latex = r"\begin{bmatrix}" + r"\\".join(rows) + r"\end{bmatrix}"

    if M.shape[0] > max_rows or M.shape[1] > max_cols:
        latex += r"\quad \text{(kısaltılmış)}"

    return latex


def poly_latex(beta):
    terms = []

    for i, c in enumerate(beta):
        sign = "+" if c >= 0 else "-"
        val = abs(c)

        if i == 0:
            term = f"{val:.5g}"
            terms.append(term if c >= 0 else f"-{term}")
        elif i == 1:
            terms.append(f" {sign} {val:.5g}x")
        else:
            terms.append(f" {sign} {val:.5g}x^{i}")

    return "".join(terms)


def parse_equation(expr, x_values):
    safe_dict = {
        "x": x_values,
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "pi": np.pi,
        "abs": np.abs
    }

    return eval(expr, {"__builtins__": {}}, safe_dict)


def ready_dataset(name):
    if name == "Basit Örnek: (1,2), (2,3), (3,5)":
        return pd.DataFrame({"x": [1, 2, 3], "y": [2, 3, 5]})

    if name == "Reklam Harcaması - Satış":
        return pd.DataFrame({"x": [2, 4, 6], "y": [15, 25, 32]})

    if name == "Ev Fiyatları":
        return pd.DataFrame({
            "x": [80, 100, 120, 140, 160],
            "y": [210, 250, 310, 330, 400]
        })

    if name == "3. Derece Deneysel Veri":
        return pd.DataFrame({
            "x": [0, 1, 2, 3, 4, 5],
            "y": [1.1, 2.9, 3.8, 6.2, 8.5, 11.0]
        })

    if name == "Radar / Dalgalı Sinyal":
        x = np.linspace(0, 10, 22)
        y = 3 * np.sin(1.5 * x) + 0.4 * x + np.random.default_rng(42).normal(0, 0.3, len(x))
        return pd.DataFrame({"x": x, "y": y})

    if name == "Ill-Conditioned Test":
        x = np.linspace(-1, 1, 24)
        y = 1 / (1 + 25 * x ** 2)
        return pd.DataFrame({"x": x, "y": y})

    return pd.DataFrame({"x": [1, 2, 3], "y": [2, 3, 5]})


# ============================================================
# BAŞLIK
# ============================================================

st.title("📐 EKK ve QR Ayrışımı Karşılaştırma Laboratuvarı")

st.markdown(
    """
<div class="glass">
Bu web uygulaması üç farklı giriş türüyle çalışır: 
<b>kullanıcının denklem girmesi</b>, <b>noktaları elle belirlemesi</b> ve 
<b>hazır veri setleriyle analiz yapması</b>. Her durumda klasik normal denklem tabanlı 
EKK yöntemi ile QR ayrışımı aynı ekranda karşılaştırılır.
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# GİRİŞ MODU
# ============================================================

st.sidebar.header("⚙️ Kullanıcı Seçenekleri")

mode = st.sidebar.radio(
    "Veri giriş türü",
    [
        "1. Kendi gireceğim denklem",
        "2. Noktaları kendim belirleyeceğim",
        "3. Hazır veri setleri"
    ]
)

degree = st.sidebar.slider("Yaklaşım polinom derecesi", 1, 15, 3)

show_ekk = st.sidebar.checkbox("EKK eğrisini göster", True)
show_qr = st.sidebar.checkbox("QR eğrisini göster", True)
show_residuals = st.sidebar.checkbox("Residual çizgilerini göster", True)
show_residual_chart = st.sidebar.checkbox("Residual bar grafiği göster", True)
scale_x = st.sidebar.checkbox("x değerlerini standartlaştır", False)

canva_link = "https://www.canva.com/design/DAHHPXzPv8s/KJexnacHpBUY-CYDl2ft4w/edit"

st.sidebar.markdown("### 🎞️ Proje Sunumu")

st.sidebar.markdown(
    f"""
    <a href="{canva_link}" target="_blank">
        <img src="qr_kod.png" style="
            width:100%;
            border-radius:12px;
            cursor:pointer;
            transition:0.2s;
        "
        onmouseover="this.style.transform='scale(1.05)'"
        onmouseout="this.style.transform='scale(1)'"
        >
    </a>
    <p style="text-align:center; font-size:13px; opacity:0.8;">
        QR'a tıkla veya telefonla okut 📱
    </p>
    """,
    unsafe_allow_html=True
)

# ============================================================
# VERİ ÜRETME
# ============================================================

st.subheader("1. Veri Giriş Alanı")

if mode == "1. Kendi gireceğim denklem":
    st.markdown("Örnek denklem: `2*x + 5`, `sin(x) + 0.2*x`, `x**3 - 2*x + 1`")

    col1, col2, col3 = st.columns(3)

    with col1:
        expr = st.text_input("f(x) =", value="sin(x) + 0.3*x")

    with col2:
        x_min = st.number_input("x başlangıç", value=0.0)

    with col3:
        x_max = st.number_input("x bitiş", value=10.0)

    col4, col5 = st.columns(2)

    with col4:
        point_count = st.slider("Nokta sayısı", 5, 100, 25)

    with col5:
        noise = st.slider("Gürültü miktarı", 0.0, 5.0, 0.3, 0.1)

    try:
        x_raw = np.linspace(x_min, x_max, point_count)
        y_clean = parse_equation(expr, x_raw)
        rng = np.random.default_rng(42)
        y_raw = y_clean + rng.normal(0, noise, len(x_raw))

        data = pd.DataFrame({"x": x_raw, "y": y_raw})
        st.dataframe(data, use_container_width=True)

    except Exception as e:
        st.error(f"Denklem okunamadı: {e}")
        st.stop()


elif mode == "2. Noktaları kendim belirleyeceğim":
    if "manual_data" not in st.session_state:
        st.session_state.manual_data = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [2.1, 3.2, 5.0, 7.4, 10.2]
        })

    data = st.data_editor(
        st.session_state.manual_data,
        num_rows="dynamic",
        use_container_width=True
    )

else:
    dataset_name = st.selectbox(
        "Hazır veri seti seç",
        [
            "Basit Örnek: (1,2), (2,3), (3,5)",
            "Reklam Harcaması - Satış",
            "Ev Fiyatları",
            "3. Derece Deneysel Veri",
            "Radar / Dalgalı Sinyal",
            "Ill-Conditioned Test"
        ]
    )

    data = ready_dataset(dataset_name)

    data = st.data_editor(
        data,
        num_rows="dynamic",
        use_container_width=True
    )


# ============================================================
# VERİ TEMİZLEME
# ============================================================

data = data.dropna().copy()
data["x"] = pd.to_numeric(data["x"], errors="coerce")
data["y"] = pd.to_numeric(data["y"], errors="coerce")
data = data.dropna().sort_values("x")

if len(data) < degree + 1:
    st.error("Seçilen derece için yeterli veri noktası yok.")
    st.stop()

x_original = data["x"].to_numpy(float)
y = data["y"].to_numpy(float)

if scale_x:
    x_mean = np.mean(x_original)
    x_std = np.std(x_original)

    if x_std == 0:
        st.error("x standart sapması sıfır olduğu için standartlaştırma yapılamaz.")
        st.stop()

    x = (x_original - x_mean) / x_std
else:
    x = x_original.copy()


# ============================================================
# HESAPLAMA
# ============================================================

A = design_matrix(x, degree)
ATA = A.T @ A

condition_A = np.linalg.cond(A)
condition_ATA = np.linalg.cond(ATA)

try:
    beta_ekk = solve_normal_equation(A, y)
    yhat_ekk = predict(beta_ekk, x)
    ekk_ok = True
except Exception:
    beta_ekk = np.full(degree + 1, np.nan)
    yhat_ekk = np.full_like(y, np.nan)
    ekk_ok = False

try:
    beta_qr, Q, R = solve_qr(A, y)
    yhat_qr = predict(beta_qr, x)
    qr_ok = True
except Exception:
    beta_qr = np.full(degree + 1, np.nan)
    yhat_qr = np.full_like(y, np.nan)
    Q = np.full_like(A, np.nan)
    R = np.full((degree + 1, degree + 1), np.nan)
    qr_ok = False

rss_ekk, rmse_ekk, mae_ekk, r2_ekk = stats(y, yhat_ekk) if ekk_ok else [np.nan] * 4
rss_qr, rmse_qr, mae_qr, r2_qr = stats(y, yhat_qr) if qr_ok else [np.nan] * 4

coef_diff = (
    np.linalg.norm(beta_ekk - beta_qr) / np.linalg.norm(beta_qr)
    if ekk_ok and qr_ok and np.linalg.norm(beta_qr) != 0
    else np.nan
)


# ============================================================
# SAYISAL KARARLILIK
# ============================================================

st.subheader("2. Sayısal Kararlılık Paneli")

c1, c2, c3, c4 = st.columns(4)

c1.metric("cond(A)", f"{condition_A:.3e}")
c2.metric("cond(AᵀA)", f"{condition_ATA:.3e}")
c3.metric("EKK - QR katsayı farkı", f"{coef_diff:.3e}")
c4.metric("Polinom derecesi", degree)

if condition_ATA > 1e12:
    st.error("Kritik uyarı: cond(AᵀA) > 10¹². Klasik EKK sonucu sayısal olarak güvenilmez olabilir.")
elif condition_ATA > 1e8:
    st.warning("Dikkat: Koşul sayısı yüksek. EKK katsayıları hassaslaşabilir.")
else:
    st.success("Koşul sayısı şu an kritik seviyenin altında.")


# ============================================================
# ANA GRAFİK
# ============================================================

st.subheader("3. EKK ve QR Eğrileri")

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
        name="Veri noktaları",
        marker=dict(size=9)
    )
)

if show_ekk and ekk_ok:
    fig.add_trace(
        go.Scatter(
            x=x_grid_original,
            y=predict(beta_ekk, x_grid),
            mode="lines",
            name="EKK / Normal Denklem",
            line=dict(width=3)
        )
    )

if show_qr and qr_ok:
    fig.add_trace(
        go.Scatter(
            x=x_grid_original,
            y=predict(beta_qr, x_grid),
            mode="lines",
            name="QR Ayrışımı",
            line=dict(width=3, dash="dash")
        )
    )

if show_residuals and qr_ok:
    for xi, yi, yqi in zip(x_original, y, yhat_qr):
        fig.add_trace(
            go.Scatter(
                x=[xi, xi],
                y=[yi, yqi],
                mode="lines",
                showlegend=False,
                hoverinfo="skip",
                line=dict(width=1, dash="dot")
            )
        )

fig.update_layout(
    template="plotly_dark",
    height=620,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,23,42,0.72)",
    title="EKK ve QR Ayrışımı Karşılaştırması",
    xaxis_title="x",
    yaxis_title="y"
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# RESIDUAL ANALİZİ
# ============================================================

if show_residual_chart:
    st.subheader("4. Residual / Kalıntı Analizi")

    residual_fig = go.Figure()

    if ekk_ok:
        residual_fig.add_trace(
            go.Bar(
                x=x_original,
                y=y - yhat_ekk,
                name="EKK kalıntıları"
            )
        )

    if qr_ok:
        residual_fig.add_trace(
            go.Bar(
                x=x_original,
                y=y - yhat_qr,
                name="QR kalıntıları"
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
        yaxis_title="Residual"
    )

    st.plotly_chart(residual_fig, use_container_width=True)


# ============================================================
# METRİK TABLOSU
# ============================================================

st.subheader("5. Hata Metrikleri")

metric_df = pd.DataFrame({
    "Metrik": ["RSS", "RMSE", "MAE", "R²"],
    "EKK": [rss_ekk, rmse_ekk, mae_ekk, r2_ekk],
    "QR": [rss_qr, rmse_qr, mae_qr, r2_qr]
})

st.dataframe(metric_df, use_container_width=True)


# ============================================================
# MATEMATİKSEL MUTFAK
# ============================================================

st.subheader("6. Matematiksel Mutfak")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["A Matrisi", "AᵀA", "Q Matrisi", "R Matrisi", "Polinomlar"]
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
    st.latex("A = " + matrix_latex(A))

with tab2:
    st.latex(
        r"""
        \hat{\beta}_{EKK} = (A^TA)^{-1}A^Ty
        """
    )
    st.latex("A^TA = " + matrix_latex(ATA))
    st.markdown(
        """
Normal denklem yöntemi teorik olarak doğru çözümü verir; ancak `AᵀA` oluşturulduğunda 
koşul sayısı büyür. Bu nedenle yüksek dereceli polinomlarda EKK çözümü kararsızlaşabilir.
"""
    )

with tab3:
    st.latex(r"A = QR, \qquad Q^TQ = I")
    st.latex("Q = " + matrix_latex(Q))

with tab4:
    st.latex(r"R\hat{\beta}_{QR}=Q^Ty")
    st.latex("R = " + matrix_latex(R))

with tab5:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### EKK Polinomu")
        if ekk_ok:
            st.latex(r"p_{EKK}(x)=" + poly_latex(beta_ekk))
            st.dataframe(
                pd.DataFrame({
                    "Derece": np.arange(len(beta_ekk)),
                    "Katsayı": beta_ekk
                }),
                use_container_width=True
            )
        else:
            st.error("EKK katsayıları hesaplanamadı.")

    with col2:
        st.markdown("### QR Polinomu")
        if qr_ok:
            st.latex(r"p_{QR}(x)=" + poly_latex(beta_qr))
            st.dataframe(
                pd.DataFrame({
                    "Derece": np.arange(len(beta_qr)),
                    "Katsayı": beta_qr
                }),
                use_container_width=True
            )
        else:
            st.error("QR katsayıları hesaplanamadı.")


# ============================================================
# SONUÇ
# ============================================================

st.subheader("7. Akademik Yorum")

st.markdown(
    f"""
<div class="glass">

Bu deneyde seçilen polinom derecesi <b>{degree}</b> için:

<br><br>

<b>cond(A)</b> = {condition_A:.3e}  
<br>
<b>cond(AᵀA)</b> = {condition_ATA:.3e}  
<br>
<b>EKK-QR göreli katsayı farkı</b> = {coef_diff:.3e}

<br><br>

Normal denklem yöntemi:

<br>

$$
A^TA\\beta = A^Ty
$$

<br>

sistemini çözer. QR yöntemi ise:

<br>

$$
A = QR, \\qquad R\\beta = Q^Ty
$$

<br>

yaklaşımını kullanır. Bu nedenle QR, özellikle kötü koşullu matrislerde daha güvenilir 
bir hesaplama yöntemi olarak öne çıkar.

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# DIŞA AKTAR
# ============================================================

st.subheader("8. Sonuçları İndir")

result_df = pd.DataFrame({
    "x": x_original,
    "y": y,
    "EKK_Tahmin": yhat_ekk,
    "QR_Tahmin": yhat_qr,
    "EKK_Residual": y - yhat_ekk,
    "QR_Residual": y - yhat_qr
})

st.download_button(
    "📥 Sonuçları CSV indir",
    data=result_df.to_csv(index=False).encode("utf-8"),
    file_name="ekk_qr_sonuclari.csv",
    mime="text/csv"
)
