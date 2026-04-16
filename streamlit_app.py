import io
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter
from scipy.stats import zscore
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ACCENT = "#58a6ff"
GREEN = "#3fb950"
ORANGE = "#d29922"
RED = "#f85149"
PURPLE = "#bc8cff"
CLUSTER_COLORS = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff", "#39d3f2"]
FEATURE_COLS = ["Log_Return", "Rolling_Vol", "Momentum_10", "ROC_5", "HL_Range", "Daily_Return"]

st.set_page_config(page_title="Time Series Mining Dashboard", layout="wide")


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
    if not set(cols).issubset(df.columns):
        raise ValueError(f"CSV must include columns: {cols}")
    out = df[cols].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out.dropna(subset=["Date", "Close"], inplace=True)
    out.sort_values("Date", inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def _parse_csv_flexible(source) -> pd.DataFrame:
    # Standard parse first.
    try:
        std = pd.read_csv(source)
        if "Date" in std.columns and "Close" in std.columns:
            if "Time" not in std.columns:
                std["Time"] = "17:00:00"
            return _ensure_columns(std)
    except Exception:
        pass

    # Retry with no header for malformed exports (e.g., second row with ticker names).
    if hasattr(source, "seek"):
        source.seek(0)
    raw = pd.read_csv(source, header=None)
    raw = raw.iloc[:, :7].copy()
    raw.columns = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
    # Remove probable header/ticker rows.
    raw = raw[~raw["Date"].astype(str).str.lower().eq("date")]
    raw = raw[~raw["Open"].astype(str).str.upper().isin(["IBM", "AAPL", "MSFT", "GOOG"])]
    return _ensure_columns(raw)


@st.cache_data(show_spinner=False)
def load_local_csv(path: str) -> pd.DataFrame:
    return _parse_csv_flexible(path)


@st.cache_data(show_spinner=False)
def load_uploaded_csv(content: bytes) -> pd.DataFrame:
    return _parse_csv_flexible(io.BytesIO(content))


@st.cache_data(show_spinner=False)
def load_yahoo_csv(ticker: str, years: int) -> pd.DataFrame:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)
    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    req = ["Open", "High", "Low", "Close", "Volume"]
    if not set(req).issubset(data.columns):
        raise ValueError("Yahoo Finance response missing OHLCV columns.")

    out = data[req].copy()
    out.reset_index(inplace=True)
    out["Time"] = "17:00:00"
    return _ensure_columns(out[["Date", "Time", "Open", "High", "Low", "Close", "Volume"]])


@st.cache_data(show_spinner=False)
def build_feature_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy().set_index("Date")
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Daily_Return"] = df["Close"].pct_change()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["Rolling_Vol"] = df["Log_Return"].rolling(20).std() * np.sqrt(252)
    df["Momentum_10"] = df["Close"].pct_change(10)
    df["ROC_5"] = df["Close"].pct_change(5)
    df["BB_Upper"] = df["SMA_20"] + 2 * df["Close"].rolling(20).std()
    df["BB_Lower"] = df["SMA_20"] - 2 * df["Close"].rolling(20).std()
    df["HL_Range"] = (df["High"] - df["Low"]) / df["Close"]
    df.dropna(inplace=True)
    return df


@st.cache_data(show_spinner=False)
def manual_stl_decompose(series: pd.Series, trend_window: int = 63):
    y = series.values.astype(float)
    wl = trend_window if trend_window % 2 == 1 else trend_window + 1
    wl = max(5, min(wl, len(y) - 1 if len(y) % 2 == 0 else len(y)))
    trend = savgol_filter(y, window_length=wl, polyorder=2)
    detrended = y - trend

    seasonal = np.zeros(len(y))
    period = 252
    for i in range(period):
        idx = np.arange(i, len(y), period)
        if len(idx):
            seasonal[idx] = np.mean(detrended[idx])

    residual = y - trend - seasonal
    return pd.Series(trend, index=series.index), pd.Series(seasonal, index=series.index), pd.Series(residual, index=series.index)


@st.cache_data(show_spinner=False)
def segment_series(df: pd.DataFrame, window: int):
    data = df[FEATURE_COLS].values
    n = len(df)
    segments, close_windows, seg_dates, seg_feature_means = [], [], [], []

    for start in range(0, n - window + 1, window):
        end = start + window
        seg = data[start:end]
        if len(seg) < window:
            continue
        segments.append(seg)
        close_windows.append(df["Close"].values[start:end])
        seg_dates.append((df.index[start], df.index[end - 1]))
        seg_feature_means.append(df[FEATURE_COLS].iloc[start:end].mean().values)

    if not segments:
        return None

    matrix = np.array([np.concatenate([seg.mean(axis=0), seg.std(axis=0)]) for seg in segments])
    matrix_scaled = StandardScaler().fit_transform(matrix)
    return np.array(segments), np.array(close_windows), matrix_scaled, seg_dates, np.array(seg_feature_means)


def dtw_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    n, m = len(s1), len(s2)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


@st.cache_data(show_spinner=False)
def compute_dtw_matrix(close_windows: np.ndarray, sample_size: int):
    if len(close_windows) == 0:
        return np.zeros((0, 0))

    np.random.seed(42)
    idx = np.sort(np.random.choice(len(close_windows), min(sample_size, len(close_windows)), replace=False))
    subs = close_windows[idx]

    normed = []
    for w in subs:
        r = w.max() - w.min()
        normed.append((w - w.min()) / r if r > 0 else np.zeros_like(w))
    normed = np.array(normed)

    n = len(normed)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(normed[i], normed[j])
            dist[i, j] = dist[j, i] = d
    return dist


@st.cache_data(show_spinner=False)
def evaluate_k(feature_matrix: np.ndarray, k_min: int, k_max: int):
    k_max = min(k_max, len(feature_matrix) - 1)
    if k_max <= k_min:
        return None, [], [], [], []

    ks = list(range(k_min, k_max + 1))
    inertias, sil_scores, db_scores = [], [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(feature_matrix)
        inertias.append(float(km.inertia_))
        sil_scores.append(float(silhouette_score(feature_matrix, labels)))
        db_scores.append(float(davies_bouldin_score(feature_matrix, labels)))

    best_k = ks[int(np.argmax(sil_scores))]
    return best_k, ks, inertias, sil_scores, db_scores


@st.cache_data(show_spinner=False)
def run_clustering(feature_matrix: np.ndarray, k: int):
    km_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(feature_matrix)
    hc_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(feature_matrix)
    return km_labels, hc_labels


@st.cache_data(show_spinner=False)
def detect_anomalies(df: pd.DataFrame, z_threshold: float, contamination: float, strong_score: int):
    res = df[["Close", "Log_Return", "Rolling_Vol", "HL_Range", "Volume"]].copy()
    res["Z_Return"] = zscore(res["Log_Return"].fillna(0))
    res["Anomaly_Z"] = res["Z_Return"].abs() > z_threshold

    q1, q3 = res["Rolling_Vol"].quantile([0.25, 0.75])
    iqr = q3 - q1
    res["Anomaly_IQR"] = (res["Rolling_Vol"] > q3 + 1.5 * iqr) | (res["Rolling_Vol"] < q1 - 1.5 * iqr)

    X = StandardScaler().fit_transform(res[["Log_Return", "Rolling_Vol", "HL_Range"]].fillna(0))
    res["Anomaly_IF"] = IsolationForest(contamination=contamination, random_state=42).fit_predict(X) == -1

    res["Score"] = res["Anomaly_Z"].astype(int) + res["Anomaly_IQR"].astype(int) + res["Anomaly_IF"].astype(int)
    res["Strong_Anomaly"] = res["Score"] >= strong_score
    return res


st.title("Time Series Mining Dashboard")

with st.sidebar:
    st.header("Data Source")
    source = st.radio("Choose source", ["Local CSV", "Upload CSV", "Yahoo Finance"], key="source_radio")

    local_candidates = [p.name for p in Path(".").glob("*.csv")]
    local_file = None
    uploaded = None
    ticker = "IBM"
    years = 5

    if source == "Local CSV":
        if not local_candidates:
            st.error("No CSV files found in this folder.")
        else:
            default_idx = local_candidates.index("IBM_5years_timeseries.csv") if "IBM_5years_timeseries.csv" in local_candidates else 0
            local_file = st.selectbox("CSV file", options=local_candidates, index=default_idx, key="local_select")
    elif source == "Upload CSV":
        uploaded = st.file_uploader("Upload a CSV", type=["csv"], key="upload_csv")
    else:
        ticker = st.text_input("Ticker", value="IBM", key="ticker_input").upper().strip()
        years = st.slider("Years", min_value=1, max_value=10, value=5, key="years_slider")

    st.header("Analysis Controls")
    window = st.slider("Segment window", 10, 60, 20, key="window_slider")
    dtw_sample = st.slider("DTW sample size", 10, 120, 60, key="dtw_sample_slider")

    c1, c2 = st.columns(2)
    with c1:
        k_min = st.number_input("K min", min_value=2, max_value=10, value=2, step=1, key="kmin_input")
    with c2:
        k_max = st.number_input("K max", min_value=3, max_value=12, value=8, step=1, key="kmax_input")

    st.header("Anomaly Controls")
    z_threshold = st.slider("Z-score threshold", 1.5, 5.0, 3.0, 0.1, key="z_slider")
    contamination = st.slider("IF contamination", 0.01, 0.15, 0.04, 0.01, key="if_slider")
    strong_score = st.slider("Strong anomaly score", 1, 3, 2, key="strong_slider")

if int(k_max) <= int(k_min):
    st.error("K max must be greater than K min.")
    st.stop()

try:
    if source == "Local CSV":
        if not local_file:
            st.stop()
        raw_df = load_local_csv(local_file)
    elif source == "Upload CSV":
        if uploaded is None:
            st.info("Upload a CSV to continue.")
            st.stop()
        raw_df = load_uploaded_csv(uploaded.getvalue())
    else:
        raw_df = load_yahoo_csv(ticker, years)
except Exception as exc:
    st.error(f"Data load failed: {exc}")
    st.stop()

try:
    df = build_feature_df(raw_df)
except Exception as exc:
    st.error(f"Feature engineering failed: {exc}")
    st.stop()

seg_result = segment_series(df, window)
if seg_result is None:
    st.error("Not enough rows for segmentation. Reduce segment window or load more data.")
    st.stop()

segments, close_windows, feature_matrix, seg_dates, seg_feature_means = seg_result
best_k, ks, inertias, sil_scores, db_scores = evaluate_k(feature_matrix, int(k_min), int(k_max))
if best_k is None:
    st.error("Not enough segments for chosen K range.")
    st.stop()

km_labels, hc_labels = run_clustering(feature_matrix, best_k)
trend, seasonal, residual = manual_stl_decompose(df["Close"])
anomaly_df = detect_anomalies(df, z_threshold, contamination, strong_score)
dtw_matrix = compute_dtw_matrix(close_windows, dtw_sample)

st.caption(f"Rows: {len(df)} | Range: {df.index.min().date()} to {df.index.max().date()} | Segments: {len(segments)} | Best K: {best_k}")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview", "Decomposition", "Clustering", "DTW and Motifs", "Anomalies", "Distribution", "Cluster Heatmap"
])

with tab1:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.42, 0.2, 0.2, 0.18])
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(color="#e6edf3", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_20"], name="SMA 20", line=dict(color=ORANGE)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="SMA 50", line=dict(color=GREEN)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_20"], name="EMA 20", line=dict(color=PURPLE, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], fill="tonexty", fillcolor="rgba(88,166,255,0.12)", line=dict(width=0), name="Bollinger"), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"] / 1e6, marker_color=ACCENT, name="Volume", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Rolling_Vol"] * 100, line=dict(color=RED), showlegend=False), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Momentum_10"] * 100, marker_color=np.where(df["Momentum_10"] >= 0, GREEN, RED), showlegend=False), row=4, col=1)
    fig.update_layout(height=860, template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig, width='stretch')

with tab2:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], line=dict(color="#e6edf3")), row=1, col=1)
    fig.add_trace(go.Scatter(x=trend.index, y=trend, line=dict(color=ACCENT)), row=2, col=1)
    fig.add_trace(go.Scatter(x=seasonal.index, y=seasonal, line=dict(color=GREEN)), row=3, col=1)
    fig.add_trace(go.Bar(x=residual.index, y=residual, marker_color=np.where(residual >= 0, GREEN, RED)), row=4, col=1)
    fig.update_layout(height=840, template="plotly_dark", showlegend=False, hovermode="x unified")
    st.plotly_chart(fig, width='stretch')

with tab3:
    c1, c2, c3 = st.columns(3)
    with c1:
        f1 = go.Figure(go.Scatter(x=ks, y=inertias, mode="lines+markers", marker=dict(color=ACCENT)))
        f1.update_layout(title="Elbow (Inertia)", template="plotly_dark", height=320)
        st.plotly_chart(f1, width='stretch')
    with c2:
        f2 = go.Figure(go.Scatter(x=ks, y=sil_scores, mode="lines+markers", marker=dict(color=GREEN)))
        f2.add_vline(x=best_k, line_dash="dash", line_color=ORANGE)
        f2.update_layout(title=f"Silhouette (best={best_k})", template="plotly_dark", height=320)
        st.plotly_chart(f2, width='stretch')
    with c3:
        f3 = go.Figure(go.Scatter(x=ks, y=db_scores, mode="lines+markers", marker=dict(color=RED)))
        f3.update_layout(title="Davies-Bouldin", template="plotly_dark", height=320)
        st.plotly_chart(f3, width='stretch')

    fp = go.Figure()
    fp.add_trace(go.Scatter(x=df.index, y=df["Close"], line=dict(color="#e6edf3", width=0.8), opacity=0.45, name="Close"))
    for (start, end), lbl in zip(seg_dates, km_labels):
        mask = (df.index >= start) & (df.index <= end)
        fp.add_trace(go.Scatter(x=df.index[mask], y=df["Close"][mask], line=dict(color=CLUSTER_COLORS[lbl % len(CLUSTER_COLORS)], width=1.8), showlegend=False))
    fp.update_layout(title="KMeans Clusters on Price", template="plotly_dark", height=420, hovermode="x unified")
    st.plotly_chart(fp, width='stretch')

    pca = PCA(n_components=2)
    proj = pca.fit_transform(feature_matrix)
    fpca = go.Figure()
    for c in range(best_k):
        mask = km_labels == c
        fpca.add_trace(go.Scatter(x=proj[mask, 0], y=proj[mask, 1], mode="markers", marker=dict(color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)], size=8), name=f"Cluster {c}"))
    fpca.update_layout(title=f"PCA (PC1={pca.explained_variance_ratio_[0]*100:.1f}%, PC2={pca.explained_variance_ratio_[1]*100:.1f}%)", template="plotly_dark", height=420)
    st.plotly_chart(fpca, width='stretch')

with tab4:
    h = go.Figure(go.Heatmap(z=dtw_matrix, colorscale="Blues"))
    h.update_layout(title="DTW Pairwise Distance Matrix", template="plotly_dark", height=500)
    st.plotly_chart(h, width='stretch')

    st.subheader("Motif Pairs by Cluster")
    for c in range(best_k):
        segs = close_windows[km_labels == c]
        if len(segs) < 2:
            continue
        normed = []
        for s in segs:
            r = s.max() - s.min()
            normed.append((s - s.min()) / r if r > 0 else np.zeros_like(s))
        normed = np.array(normed)

        best_dist, bi, bj = np.inf, 0, 1
        cap = min(len(normed), 25)
        for i in range(cap):
            for j in range(i + 1, cap):
                d = dtw_distance(normed[i], normed[j])
                if d < best_dist:
                    best_dist, bi, bj = d, i, j

        fm = go.Figure()
        fm.add_trace(go.Scatter(y=normed[bi], line=dict(color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)], width=2), name="Segment A"))
        fm.add_trace(go.Scatter(y=normed[bj], line=dict(color="#ffffff", width=2, dash="dash"), name="Segment B"))
        fm.update_layout(title=f"Cluster {c} Motif | DTW={best_dist:.3f}", template="plotly_dark", height=320)
        st.plotly_chart(fm, width='stretch')

with tab5:
    strong = anomaly_df[anomaly_df["Strong_Anomaly"]]
    fa = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    fa.add_trace(go.Scatter(x=df.index, y=df["Close"], line=dict(color="#e6edf3"), name="Close"), row=1, col=1)
    fa.add_trace(go.Scatter(x=strong.index, y=df.loc[strong.index, "Close"], mode="markers", marker=dict(color=RED, size=7), name="Strong anomaly"), row=1, col=1)
    fa.add_trace(go.Bar(x=anomaly_df.index, y=anomaly_df["Z_Return"], marker_color=np.where(anomaly_df["Z_Return"].abs() > z_threshold, RED, ACCENT), name="Z-score"), row=2, col=1)
    fa.add_trace(go.Scatter(x=anomaly_df.index, y=anomaly_df["Rolling_Vol"] * 100, line=dict(color=PURPLE), name="Volatility"), row=3, col=1)
    fa.update_layout(height=820, template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fa, width='stretch')

with tab6:
    c1, c2, c3 = st.columns(3)
    with c1:
        fh = go.Figure(go.Histogram(x=df["Log_Return"] * 100, nbinsx=60, marker_color=ACCENT))
        fh.update_layout(title="Log Return Distribution", template="plotly_dark", height=320)
        st.plotly_chart(fh, width='stretch')
    with c2:
        fv = go.Figure(go.Scatter(x=df.index, y=df["Rolling_Vol"] * 100, line=dict(color=RED), fill="tozeroy"))
        fv.update_layout(title="Rolling 20-day Volatility", template="plotly_dark", height=320)
        st.plotly_chart(fv, width='stretch')
    with c3:
        sorted_ret = np.sort(df["Log_Return"].dropna()) * 100
        n = len(sorted_ret)
        theo = np.array([np.percentile(np.random.normal(0, 1, 10000), 100 * (i + 0.5) / n) for i in range(n)]) * df["Log_Return"].std() * 100
        fqq = go.Figure()
        fqq.add_trace(go.Scatter(x=theo, y=sorted_ret, mode="markers", marker=dict(color=PURPLE, size=5), name="Sample"))
        fqq.add_trace(go.Scatter(x=[theo.min(), theo.max()], y=[theo.min(), theo.max()], mode="lines", line=dict(color=ORANGE), name="Normal line"))
        fqq.update_layout(title="Q-Q Plot", template="plotly_dark", height=320)
        st.plotly_chart(fqq, width='stretch')

with tab7:
    cluster_rows = []
    cluster_names = []
    for c in range(best_k):
        rows = seg_feature_means[km_labels == c]
        if len(rows) == 0:
            continue
        cluster_rows.append(rows.mean(axis=0))
        cluster_names.append(f"C{c}")

    if not cluster_rows:
        st.warning("Unable to build cluster heatmap.")
    else:
        hmap = pd.DataFrame(cluster_rows, columns=FEATURE_COLS, index=cluster_names)
        hmap_scaled = (hmap - hmap.mean()) / (hmap.std() + 1e-8)
        fhm = go.Figure(go.Heatmap(z=hmap_scaled.values, x=hmap_scaled.columns, y=hmap_scaled.index, colorscale="RdYlGn", zmid=0))
        fhm.update_layout(title="Cluster Feature Profiles (standardized)", template="plotly_dark", height=380)
        st.plotly_chart(fhm, width='stretch')

