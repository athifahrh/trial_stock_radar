# ==============================================
# 📊 Streamlit – Stock Radar OLD (Swing & Short-Term)
# Full IDX auto (GitHub) — no API key
# ==============================================

import hashlib
import requests
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go


# -----------------------------
# Page config + CSS
# -----------------------------
st.set_page_config(page_title="Stock Radar OLD – Swing & Short-Term", page_icon="📊", layout="wide")

def load_css():
    here = Path(__file__).resolve()
    for p in [here.parent / "style.css", here.parent.parent / "style.css", Path.cwd() / "style.css"]:
        if p.exists():
            st.markdown(f"<style>{p.read_text()}</style>", unsafe_allow_html=True)
            return

load_css()


# -----------------------------
# Utils
# -----------------------------
def stable_shuffle(symbols: list[str]) -> list[str]:
    return sorted(
        symbols,
        key=lambda s: hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest()
    )

def normalize_ticker(s: str) -> str:
    s = str(s).strip().upper()
    return s if s.endswith(".JK") else f"{s}.JK"


# -----------------------------
# Full IDX universe via GitHub (no API key)
# -----------------------------
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def fetch_idx_symbols_github() -> list[str]:
    """
    Pull list from wildangunawan/Dataset-Saham-IDX -> Saham/Semua (filenames .csv = tickers)
    """
    owner, repo, path = "wildangunawan", "Dataset-Saham-IDX", "Saham/Semua"
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()

    out = []
    for it in js:
        name = (it.get("name") or "").strip()
        if not name.lower().endswith(".csv"):
            continue
        sym = name[:-4].upper()
        if sym:
            out.append(f"{sym}.JK")

    if not out:
        raise ValueError("No tickers parsed from GitHub listing.")

    out = list(dict.fromkeys(out))  # dedupe keep order
    return stable_shuffle(out)


# -----------------------------
# Indicator helpers
# -----------------------------
def ema(s, n): 
    return s.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series, n=20, k=2):
    ma = series.rolling(n).mean()
    std = series.rolling(n).std()
    upper, lower = ma + k*std, ma - k*std
    width = (upper - lower) / ma
    return ma, upper, lower, width

def atr(df, n=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def fib_levels(series, lookback=100):
    win = series.tail(lookback)
    hi, lo = win.max(), win.min()
    diff = hi - lo
    return {
        "23.6": hi - 0.236 * diff,
        "38.2": hi - 0.382 * diff,
        "50.0": hi - 0.5 * diff,
        "61.8": hi - 0.618 * diff,
        "78.6": hi - 0.786 * diff,
        "high": hi,
        "low": lo
    }


# -----------------------------
# Scoring (same logic as your OLD)
# -----------------------------
def score_stock(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0

    c = df["Close"]
    v = df["Volume"]

    ema20, ema50, ema200 = ema(c, 20), ema(c, 50), ema(c, 200)
    macd_line, sig, hist = macd(c)
    r = rsi(c)
    bb_ma, bb_up, bb_low, _ = bollinger(c)

    score = 0.0
    # discrete points
    if ema20.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1]:
        score += 40
    if c.iloc[-1] > ema20.iloc[-1]:
        score += 10
    if macd_line.iloc[-1] > sig.iloc[-1]:
        score += 20
    if len(hist) >= 2 and hist.iloc[-1] > hist.iloc[-2]:
        score += 10
    if 40 <= r.iloc[-1] <= 60:
        score += 10
    if len(v) >= 20 and v.iloc[-1] > 1.5 * v.rolling(20).mean().iloc[-1]:
        score += 10

    # continuous tie-break additives
    ema_gap = float(c.iloc[-1] / (ema20.iloc[-1] + 1e-9) - 1.0)
    macd_slope = float(macd_line.diff().iloc[-1]) if len(macd_line) >= 2 else 0.0
    rsi_center = float(1.0 - abs(r.iloc[-1] - 50.0) / 50.0)
    vol_ratio = float(v.iloc[-1] / (v.rolling(20).mean().iloc[-1] + 1e-9)) if len(v) >= 20 else 0.0
    bb_span = float((bb_up.iloc[-1] - bb_low.iloc[-1]) / (bb_ma.iloc[-1] + 1e-9)) if np.isfinite(bb_ma.iloc[-1]) else 0.0
    bb_pos = float((c.iloc[-1] - bb_low.iloc[-1]) / ((bb_up.iloc[-1] - bb_low.iloc[-1]) + 1e-9))

    score += (
        5.0 * ema_gap +
        3.0 * macd_slope +
        2.0 * rsi_center +
        2.0 * np.log1p(max(vol_ratio - 1.0, 0.0)) +
        1.0 * bb_pos +
        1.0 * bb_span
    )

    return round(float(score), 3)


# -----------------------------
# Batched download (1d)
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices_batched(tickers: list[str], period="3mo", batch_size=30) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    n = len(tickers)

    for i in range(0, n, batch_size):
        batch = tickers[i:i + batch_size]
        data = yf.download(
            " ".join(batch),
            period=period, interval="1d",
            progress=False, auto_adjust=False,
            group_by="ticker", threads=True
        )
        if data is None or data.empty:
            continue

        if isinstance(data.columns, pd.MultiIndex):
            for t in batch:
                if t in data.columns.levels[0]:
                    df = data[t].dropna()
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        out[t] = df
        else:
            if len(batch) == 1 and not data.empty:
                out[batch[0]] = data.dropna()

    return out


def rolling_avg_vol(df: pd.DataFrame, w: int = 20) -> float:
    try:
        if df is None or df.empty or len(df) < w:
            return 0.0
        return float(df["Volume"].rolling(w).mean().iloc[-1])
    except Exception:
        return 0.0


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Scanner Settings (Full IDX)")

    period = st.selectbox("Data period", ["3mo", "6mo", "1y"], index=0)
    max_names = st.slider("Max tickers to scan", 50, 900, 300, step=50)
    batch_size = st.slider("Batch size (downloads)", 20, 120, 30, step=10)

    vol_window = st.selectbox("Rolling volume window (days)", [10, 20, 30], index=1)
    min_avg_vol = st.number_input(f"Min avg volume (rolling {vol_window}d)", value=10_000_000, step=100_000)

    show_top_n = st.slider("Show top N", 20, 200, 50, step=10)

    refresh = st.button("🔄 Refresh data", use_container_width=True)
    if refresh:
        fetch_prices_batched.clear()
        fetch_idx_symbols_github.clear()


# -----------------------------
# Header
# -----------------------------
st.title("📊 Stock Radar OLD — Swing & Short-Term (Full IDX)")
st.markdown("<hr class='hr-soft'>", unsafe_allow_html=True)
st.markdown("""
<div class='card'>
<h4>📘 Overview</h4>
Versi OLD untuk screening swing/short-term berbasis <b>EMA alignment, MACD, RSI, Bollinger</b> + volume.
Universe: <b>Full IDX otomatis</b> dari GitHub.
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Universe (Full IDX)
# -----------------------------
with st.spinner("Pulling Full IDX universe from GitHub..."):
    universe = fetch_idx_symbols_github()
universe = [normalize_ticker(t) for t in universe]
st.caption(f"Universe (Full IDX) size: **{len(universe)}**")

universe = universe[:int(max_names)]
st.caption(f"Universe size in this run: **{len(universe)}**")


# -----------------------------
# Fetch prices
# -----------------------------
with st.spinner("Fetching prices..."):
    price_map = fetch_prices_batched(universe, period=period, batch_size=int(batch_size))


# -----------------------------
# Build ranking
# -----------------------------
rows = []
for t, df_t in price_map.items():
    try:
        if df_t is None or df_t.empty:
            continue
        if len(df_t) < 60:  # biar EMA200 ga ngaco, minimal data cukup
            continue

        avgv = rolling_avg_vol(df_t, w=int(vol_window))
        if avgv < float(min_avg_vol):
            continue

        c = df_t["Close"]
        v = df_t["Volume"]

        score = float(score_stock(df_t))
        mom20 = float(c.pct_change(20).iloc[-1]) if len(c) >= 21 else np.nan
        volr = float(v.iloc[-1] / (v.rolling(20).mean().iloc[-1] + 1e-9)) if len(v) >= 20 else np.nan

        tb_raw = hashlib.blake2b(t.encode("utf-8"), digest_size=4).digest()
        tiebreak = int.from_bytes(tb_raw, "big") / (256**4 - 1)

        rows.append({
            "ticker": t,
            "score": score,
            "last": float(c.iloc[-1]),
            "mom20": mom20,
            "volr": volr,
            "avg_vol": avgv,
            "tiebreak": tiebreak,
        })
    except Exception:
        pass

rank = pd.DataFrame(rows)

if rank.empty:
    st.warning("No stocks passed the filters. Coba turunin min_avg_vol / kecilkan vol_window / besarin max_names.")
    st.stop()

# ensure numeric
for col in ["score", "mom20", "volr", "last", "avg_vol", "tiebreak"]:
    rank[col] = pd.to_numeric(rank[col], errors="coerce")

rank["mom20"] = rank["mom20"].fillna(-1e9)
rank["volr"] = rank["volr"].fillna(-1e9)

rank = (rank
        .dropna(subset=["score"])
        .drop_duplicates(subset="ticker", keep="last")
        .sort_values(["score", "mom20", "volr", "tiebreak"], ascending=[False, False, False, False])
        .reset_index(drop=True))

top = rank.head(int(show_top_n)).copy()

st.markdown("## 📌 Stocks Recommendation")
st.dataframe(
    top.assign(
        score=top["score"].map(lambda x: f"{x:.0f}"),
        last=top["last"].map(lambda x: f"{x:,.2f}"),
        mom20=top["mom20"].map(lambda x: f"{x*100:.2f}%" if np.isfinite(x) else ""),
        volr=top["volr"].map(lambda x: f"{x:.2f}" if np.isfinite(x) else ""),
        avg_vol=top["avg_vol"].map(lambda x: f"{x:,.0f}"),
    ),
    use_container_width=True,
    hide_index=True
)


# -----------------------------
# Inspector
# -----------------------------
st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)
st.markdown("## 📈 Inspect Stock")

pick = st.selectbox("Select stock", options=top["ticker"].tolist(), index=0)
df = price_map.get(pick, pd.DataFrame())

if df is None or df.empty:
    st.warning("No data for selected stock.")
    st.stop()

close = df["Close"]
ema20, ema50, ema200 = ema(close, 20), ema(close, 50), ema(close, 200)
bb_ma, bb_up, bb_low, _ = bollinger(close)
fib = fib_levels(close, 100)
a = atr(df)

TITLE_STYLE = dict(
    x=0.5,
    xanchor="center",
    font=dict(size=18, color="#0b1220", family="sans-serif"),
)

# Chart 1: Price
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="Price"
))
fig.add_trace(go.Scatter(x=df.index, y=ema20, name="EMA20", mode="lines"))
fig.add_trace(go.Scatter(x=df.index, y=ema50, name="EMA50", mode="lines"))
fig.add_trace(go.Scatter(x=df.index, y=ema200, name="EMA200", mode="lines"))
fig.add_trace(go.Scatter(x=df.index, y=bb_up, name="BB Upper", mode="lines", line=dict(width=1)))
fig.add_trace(go.Scatter(x=df.index, y=bb_low, name="BB Lower", mode="lines", line=dict(width=1)))

for k, v in fib.items():
    if k not in {"high", "low"}:
        fig.add_hline(y=v, line_dash="dot", annotation_text=f"Fib {k}%")

fig.update_layout(
    title=dict(text=f"📈 Price Action – {pick}", **TITLE_STYLE),
    height=520,
    margin=dict(l=0, r=0, t=60, b=0),
    xaxis_rangeslider_visible=False
)
st.plotly_chart(fig, use_container_width=True)

# Chart 2: MACD
macd_line, sig, hist = macd(close)
macd_fig = go.Figure()
macd_fig.add_trace(go.Scatter(x=df.index, y=macd_line, name="MACD"))
macd_fig.add_trace(go.Scatter(x=df.index, y=sig, name="Signal"))
macd_fig.add_trace(go.Bar(x=df.index, y=hist, name="Hist"))
macd_fig.update_layout(
    title=dict(text=f"📊 MACD Momentum – {pick}", **TITLE_STYLE),
    height=240,
    margin=dict(l=0, r=0, t=60, b=0)
)
st.plotly_chart(macd_fig, use_container_width=True)

# Chart 3: RSI
r = rsi(close)
rsi_fig = go.Figure()
rsi_fig.add_trace(go.Scatter(x=df.index, y=r, name="RSI"))
rsi_fig.add_hrect(y0=40, y1=60, fillcolor="LightGray", opacity=0.3, line_width=0)
rsi_fig.update_layout(
    title=dict(text=f"📉 RSI – {pick}", **TITLE_STYLE),
    height=220,
    margin=dict(l=0, r=0, t=60, b=0)
)
st.plotly_chart(rsi_fig, use_container_width=True)

# Risk / Reward
last_close = float(close.iloc[-1])
atr_last = float(a.iloc[-1]) if np.isfinite(a.iloc[-1]) else 0.0
stop = last_close - 1.5 * atr_last
target = last_close + 2.0 * atr_last
rr = (target - last_close) / (last_close - stop + 1e-9)

st.markdown(f"""
<div style='height:10px;'></div>
<div class='metrics-3'>
  <div class='metric'><div class='k'>💰 Last Close</div><div class='v'>{last_close:,.2f}</div></div>
  <div class='metric'><div class='k'>🎯 Target</div><div class='v'>{target:,.2f}</div></div>
  <div class='metric'><div class='k'>🛑 Stop Loss</div><div class='v'>{stop:,.2f}</div></div>
</div>
<div class='subtle-subtitle' style='text-align:center; font-size:1rem; margin-top:8px;'>
⚖️ Risk–Reward Ratio ≈ 1 : {rr:,.2f}
</div>
<div style='height:25px;'></div>
""", unsafe_allow_html=True)
