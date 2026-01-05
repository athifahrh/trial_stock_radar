# ==========================================
# 🧠 Stock Radar NEW — Dynamic Only (Daily → 5m)
# Universe: Full IDX (GitHub) -> Daily prefilter -> 5m scan
# No manual upload, no EMA/MACD/RSI
# Score normalized to 0..100 (FIX)
# Daily volume baseline window selectable (3/5/10/20)
# + Ranking modes: Flow / Scalp / Swing
# ==========================================

import hashlib
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Stock Radar NEW (5m) — Dynamic", page_icon="📊", layout="wide")

# -----------------------------
# Utils
# -----------------------------
def stable_shuffle(symbols: list[str]) -> list[str]:
    return sorted(
        symbols,
        key=lambda s: hashlib.blake2b(s.encode("utf-8"), digest_size=4).digest()
    )

def clamp(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))

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

    out = list(dict.fromkeys(out))
    return stable_shuffle(out)

# -----------------------------
# yfinance fetchers
# -----------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_daily_batched(tickers: list[str], period="6mo", batch_size=200) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        data = yf.download(
            " ".join(batch),
            period=period, interval="1d",
            progress=False, auto_adjust=False,
            group_by="ticker", threads=False
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
            if len(batch) == 1:
                out[batch[0]] = data.dropna()
    return out

@st.cache_data(ttl=300, show_spinner=False)
def fetch_5m_batched(tickers: list[str], period="5d", batch_size=30) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        data = yf.download(
            " ".join(batch),
            period=period, interval="5m",
            progress=False, auto_adjust=False,
            group_by="ticker", threads=False
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
            if len(batch) == 1:
                out[batch[0]] = data.dropna()
    return out

# -----------------------------
# Daily prefilter (Option B) - window selectable
# -----------------------------
def daily_prefilter(daily_map: dict[str, pd.DataFrame], min_avg_vol: float, top_n: int, vol_window: int) -> pd.DataFrame:
    rows = []
    w = int(vol_window)

    for t, df in daily_map.items():
        try:
            if df is None or df.empty or len(df) < (w + 2):
                continue

            v = df["Volume"].astype(float)
            c = df["Close"].astype(float)

            avgN = float(v.rolling(w, min_periods=w).mean().iloc[-1])
            if not np.isfinite(avgN) or avgN <= 0:
                continue
            if avgN < float(min_avg_vol):
                continue

            rel_vol = float(v.iloc[-1] / (avgN + 1e-9))
            ret1d = float(c.pct_change(1).iloc[-1])

            life = (1.4 * clamp((rel_vol - 1.0) / 2.5)) + (0.6 * clamp(abs(ret1d) / 0.05))

            rows.append({
                "ticker": t,
                f"avg{w}_vol": avgN,
                "avg_vol": avgN,
                "rel_vol": rel_vol,
                "ret1d": ret1d,
                "life": float(life),
            })
        except Exception:
            continue

    cand = pd.DataFrame(rows)
    if cand.empty:
        return cand

    cand = (cand
            .sort_values(["life", "rel_vol"], ascending=[False, False])
            .head(int(top_n))
            .reset_index(drop=True))
    return cand

# -----------------------------
# 5m helpers
# -----------------------------
def intraday_groups(df5: pd.DataFrame) -> dict:
    if df5 is None or df5.empty:
        return {}
    tmp = df5.copy()
    tmp["d"] = tmp.index.date
    out = {}
    for d, sub in tmp.groupby("d"):
        out[d] = sub.drop(columns=["d"])
    return out

def hap_proxy_events(day_df: pd.DataFrame, lookback_bars=12) -> tuple[float, pd.DataFrame]:
    if day_df is None or day_df.empty or len(day_df) < lookback_bars + 2:
        return 0.0, pd.DataFrame()

    df = day_df.copy()
    df["roll_high"] = df["High"].shift(1).rolling(lookback_bars).max()
    df["bar_range"] = (df["High"] - df["Low"]).replace(0, np.nan)
    df["close_pos_bar"] = (df["Close"] - df["Low"]) / (df["bar_range"] + 1e-9)
    df["vol_med"] = df["Volume"].rolling(lookback_bars).median()

    df["is_break"] = (df["Close"] > df["roll_high"]) & df["roll_high"].notna()
    df["vol_burst"] = df["Volume"] > (2.5 * df["vol_med"].fillna(0))
    df["close_strong"] = df["close_pos_bar"] >= 0.70
    df["hap_bar"] = df["is_break"] & df["vol_burst"] & df["close_strong"]

    events = df[df["hap_bar"]].copy()
    if events.empty:
        return 0.0, pd.DataFrame()

    burst_mag = (events["Volume"] / (events["vol_med"] + 1e-9)).clip(0, 10)
    raw = float(events.shape[0]) * float(burst_mag.mean())
    score = 20.0 * float(np.tanh(raw / 6.0))

    out = events[["Open", "High", "Low", "Close", "Volume", "roll_high", "vol_med", "close_pos_bar"]].copy()
    out["burst_x"] = (events["Volume"] / (events["vol_med"] + 1e-9)).round(2)
    out["close_pos_bar"] = out["close_pos_bar"].round(2)
    return score, out

def score_from_5m_and_daily(df5: pd.DataFrame, daily_row: pd.Series) -> tuple[dict, pd.DataFrame]:
    days = intraday_groups(df5)
    if not days:
        return {}, pd.DataFrame()

    scan_date = sorted(days.keys())[-1]
    day_df = days.get(scan_date)
    if day_df is None or day_df.empty:
        return {}, pd.DataFrame()

    o = float(day_df["Open"].iloc[0])
    h = float(day_df["High"].max())
    l = float(day_df["Low"].min())
    c = float(day_df["Close"].iloc[-1])
    rng = max(h - l, 1e-9)

    close_pos = float((c - l) / rng)
    ret_day = float((c / (o + 1e-9)) - 1.0)

    day_vol = float(day_df["Volume"].sum())
    early_vol = float(day_df["Volume"].iloc[:6].sum()) if len(day_df) >= 6 else day_vol

    prev_day_vol = np.nan
    sorted_days = sorted(days.keys())
    idx = sorted_days.index(scan_date)
    if idx > 0:
        prev_day_vol = float(days[sorted_days[idx - 1]]["Volume"].sum())
    early_vs_prev = float(early_vol / (prev_day_vol + 1e-9)) if np.isfinite(prev_day_vol) and prev_day_vol > 0 else np.nan

    rel_vol = float(daily_row.get("rel_vol", np.nan))
    ret1d = float(daily_row.get("ret1d", np.nan))

    # Activity (0..25)
    a = clamp((rel_vol - 0.9) / (3.0 - 0.9)) if np.isfinite(rel_vol) else 0.0
    if np.isfinite(early_vs_prev):
        e = clamp((early_vs_prev - 0.15) / (0.50 - 0.15))
        a = 0.70 * a + 0.30 * e
    S_activity = 25.0 * a

    # VPA (0..25)
    cp = clamp((close_pos - 0.5) / 0.5)
    abs_ret = abs(ret_day)
    vol_term = clamp((rel_vol - 1.0) / 2.5) if np.isfinite(rel_vol) else 0.0

    mid, spread = 0.012, 0.015
    abs_term = float(np.exp(-((abs_ret - mid) ** 2) / (2 * spread ** 2))) if np.isfinite(abs_ret) else 0.0
    absorp = 0.55 * vol_term + 0.45 * abs_term
    S_vpa = 25.0 * (0.60 * cp + 0.40 * absorp)

    # HAP (0..20)
    S_hap, hap_events = hap_proxy_events(day_df, lookback_bars=12)

    raw = 0.45 * S_activity + 0.40 * S_vpa + 0.15 * S_hap
    RAW_MAX = 24.25
    score = float((raw / RAW_MAX) * 100.0)

    tb = hashlib.blake2b(str(daily_row.get("ticker", "")).encode("utf-8"), digest_size=4).digest()
    tiebreak = int.from_bytes(tb, "big") / (256**4 - 1)

    metrics = {
        "scan_date": str(scan_date),
        "score": float(score),
        "S_activity": float(S_activity),
        "S_vpa": float(S_vpa),
        "S_hap": float(S_hap),
        "rel_vol": float(rel_vol) if np.isfinite(rel_vol) else np.nan,
        "ret1d": float(ret1d) if np.isfinite(ret1d) else np.nan,
        "ret_day": float(ret_day),
        "close_pos": float(close_pos),
        "early_vs_prev": float(early_vs_prev) if np.isfinite(early_vs_prev) else np.nan,
        "last_close": float(c),
        "day_vol_5m_sum": float(day_vol),
        "tiebreak": float(tiebreak),
    }
    return metrics, hap_events

def scalp_score_intraday_5m(day_df: pd.DataFrame) -> dict:
    if day_df is None or day_df.empty or len(day_df) < 10:
        return {"scalp_score": np.nan}

    df = day_df.copy()

    o = float(df["Open"].iloc[0])
    h = float(df["High"].max())
    l = float(df["Low"].min())
    c = float(df["Close"].iloc[-1])
    rng = max(h - l, 1e-9)
    close_pos = (c - l) / rng

    early_bars = min(6, len(df))
    early_vol = float(df["Volume"].iloc[:early_bars].sum())
    total_vol = float(df["Volume"].sum())
    rest_vol = max(total_vol - early_vol, 1e-9)
    early_ratio = early_vol / (rest_vol + 1e-9)

    up_mask = (df["Close"] > df["Open"])
    down_mask = (df["Close"] < df["Open"])
    up_vol = float(df.loc[up_mask, "Volume"].sum())
    down_vol = float(df.loc[down_mask, "Volume"].sum())
    up_dom = up_vol / (up_vol + down_vol + 1e-9)

    tail = df.tail(min(6, len(df)))
    tail_min_close = float(tail["Close"].min())
    dd_tail = (c - tail_min_close) / (c + 1e-9)

    S_early = clamp((early_ratio - 0.20) / (1.20 - 0.20))
    S_updom = clamp((up_dom - 0.45) / (0.70 - 0.45))
    S_hold  = 0.65 * clamp((close_pos - 0.55) / (0.90 - 0.55)) + 0.35 * clamp((0.020 - dd_tail) / 0.020)

    scalp = 100.0 * (0.40 * S_early + 0.35 * S_updom + 0.25 * S_hold)

    return {
        "scalp_score": float(scalp),
        "scalp_early_ratio": float(early_ratio),
        "scalp_up_dom": float(up_dom),
        "scalp_close_pos": float(close_pos),
        "scalp_dd_tail": float(dd_tail),
    }

def swing_score_daily(dfD: pd.DataFrame) -> dict:
    if dfD is None or dfD.empty or len(dfD) < 25:
        return {"swing_score": np.nan}

    df = dfD.copy()
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)

    ret10 = float(c.pct_change(10).iloc[-1]) if len(c) >= 11 else np.nan
    ret20 = float(c.pct_change(20).iloc[-1]) if len(c) >= 21 else np.nan

    hi20 = float(h.rolling(20).max().iloc[-1])
    lo20 = float(l.rolling(20).min().iloc[-1])
    rng20 = max(hi20 - lo20, 1e-9)
    close_pos_20 = float((c.iloc[-1] - lo20) / rng20)

    avg20v = float(v.rolling(20).mean().iloc[-1])
    volr = float(v.iloc[-1] / (avg20v + 1e-9))

    ret1 = float(c.pct_change(1).iloc[-1])
    vol10 = float(c.pct_change().rolling(10).std().iloc[-1]) if len(c) >= 12 else np.nan

    S_trend = 0.55 * clamp((ret10 - 0.00) / (0.12 - 0.00)) + 0.45 * clamp((ret20 - 0.00) / (0.20 - 0.00))
    S_strength = clamp((close_pos_20 - 0.55) / (0.95 - 0.55))
    S_break = 0.55 * clamp((ret1 - 0.002) / (0.03 - 0.002)) + 0.45 * clamp((volr - 1.0) / (2.5 - 1.0))

    if np.isfinite(vol10):
        S_stable = clamp((0.050 - vol10) / (0.050 - 0.010))
    else:
        S_stable = 0.5

    swing = 100.0 * (0.40 * S_trend + 0.30 * S_strength + 0.20 * S_break + 0.10 * S_stable)

    return {
        "swing_score": float(swing),
        "swing_ret10": float(ret10) if np.isfinite(ret10) else np.nan,
        "swing_ret20": float(ret20) if np.isfinite(ret20) else np.nan,
        "swing_close_pos_20": float(close_pos_20),
        "swing_volr": float(volr),
        "swing_vol10": float(vol10) if np.isfinite(vol10) else np.nan,
    }

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Scanner Settings (Dynamic Only)")

    rank_mode = st.radio(
        "Ranking mode",
        ["Flow (Volume Flow Score)", "Scalp (Intraday Breakout)", "Swing (Multi-day Trend)"],
        index=0
    )
    min_scalp = st.slider("Min scalp score (0..100)", 0, 100, 55, step=1)
    min_swing = st.slider("Min swing score (0..100)", 0, 100, 55, step=1)


    intraday_period = st.selectbox("Data period (for 5m)", ["5d", "10d"], index=0)
    intraday_batch = st.slider("5m batch size", 10, 120, 30, step=10)

    top_n_daily = st.slider("Daily prefilter: top N candidates", 50, 600, 250, step=50)
    daily_batch = st.slider("Daily batch size", 100, 500, 200, step=50)

    vol_window = st.selectbox("Daily volume baseline window", [3, 5, 10, 20], index=3)
    min_avg_vol = st.number_input(f"Min avg DAILY volume (rolling {vol_window}d)", value=10_000_000, step=100_000)

    min_score = st.slider("Min flow score (0..100)", 0, 100, 55, step=1)

    refresh = st.button("🔄 Refresh data", use_container_width=True)
    if refresh:
        fetch_daily_batched.clear()
        fetch_5m_batched.clear()
        fetch_idx_symbols_github.clear()

# -----------------------------
# Header
# -----------------------------
st.title("📊 Stock Radar")

# -----------------------------
# Universe
# -----------------------------
with st.spinner("Pulling Full IDX universe from GitHub..."):
    base_universe = fetch_idx_symbols_github()
base_universe = [normalize_ticker(t) for t in base_universe]
st.caption(f"Universe (Full IDX) size: **{len(base_universe)}**")

# -----------------------------
# Stage 1: Daily prefilter
# -----------------------------
st.markdown("### 1) Daily prefilter")

with st.spinner("Fetching DAILY data (batch)..."):
    daily_map = fetch_daily_batched(base_universe, period="6mo", batch_size=int(daily_batch))

st.write(f"Daily data available: **{len(daily_map)}** tickers")

cand = daily_prefilter(
    daily_map,
    min_avg_vol=float(min_avg_vol),
    top_n=int(top_n_daily),
    vol_window=int(vol_window)
)

if cand.empty:
    st.warning("Tidak ada kandidat lolos daily prefilter. Turunin Min avg volume atau klik Refresh.")
    st.stop()

# add swing metrics per candidate (daily only)
swing_rows = []
for t in cand["ticker"].tolist():
    dfD = daily_map.get(t)
    s = swing_score_daily(dfD)
    s["ticker"] = t
    swing_rows.append(s)

swing_df = pd.DataFrame(swing_rows)
if not swing_df.empty:
    cand = cand.merge(swing_df, on="ticker", how="left")

display = cand.copy()
display["avg_vol_disp"] = display["avg_vol"].round(0)
display["rel_vol_disp"] = display["rel_vol"].round(2)
display["ret1d_disp"] = (display["ret1d"] * 100).round(2)
display["life_disp"] = display["life"].round(2)
display["swing_score_disp"] = display["swing_score"].round(1)

st.dataframe(
    display[[
        "ticker",
        "avg_vol_disp",
        "rel_vol_disp",
        "ret1d_disp",
        "life_disp",
        "swing_score_disp"
    ]].rename(columns={
        "avg_vol_disp": "avg_vol",
        "rel_vol_disp": "rel_vol",
        "ret1d_disp": "ret1d (%)",
        "life_disp": "life",
        "swing_score_disp": "swing_score"
    }),
    use_container_width=True,
    hide_index=True
)

cand_lookup = {r["ticker"]: r for _, r in cand.iterrows()}
scan_universe = cand["ticker"].tolist()

# -----------------------------
# Stage 2: 5m scan
# -----------------------------
st.markdown("### 2) 5m scan")
st.caption(f"Tickers going into 5m fetch: **{len(scan_universe)}**")

with st.spinner("Fetching 5m data (batch)..."):
    price_map_5m = fetch_5m_batched(scan_universe, period=intraday_period, batch_size=int(intraday_batch))

st.write(f"5m data available: **{len(price_map_5m)}** tickers")

rows = []
hap_store: dict[str, pd.DataFrame] = {}

SWING_COLS = ["swing_score", "swing_ret10", "swing_ret20", "swing_close_pos_20", "swing_volr", "swing_vol10"]

for t, df5 in price_map_5m.items():
    try:
        daily_row = cand_lookup.get(t, pd.Series({"ticker": t, "rel_vol": np.nan, "ret1d": np.nan}))

        metrics_flow, hap_events = score_from_5m_and_daily(df5, daily_row)
        if not metrics_flow:
            continue

        days = intraday_groups(df5)
        if not days:
            continue
        scan_date = sorted(days.keys())[-1]
        day_df = days.get(scan_date)
        if day_df is None or day_df.empty:
            continue

        scalp = scalp_score_intraday_5m(day_df)

        out = dict(metrics_flow)
        out["ticker"] = t
        out.update(scalp)

        # ✅ inject swing columns from daily_row into rank output (FIX)
        for col in SWING_COLS:
            out[col] = float(daily_row.get(col, np.nan)) if col in daily_row else np.nan

        rows.append(out)
        hap_store[t] = hap_events
    except Exception:
        continue

rank = pd.DataFrame(rows)
if rank.empty:
    st.warning(
        "Hasil kosong di 5m stage. Penyebab umum:\n"
        "- Data 5m untuk .JK lagi bolong\n"
        "- Hari ini data intraday belum kebaca\n"
        "Coba Refresh atau ganti period 10d."
    )
    st.stop()

# ✅ guard: create missing swing cols (extra safety)
for col in SWING_COLS:
    if col not in rank.columns:
        rank[col] = np.nan

# ensure numeric
rank["score"] = pd.to_numeric(rank["score"], errors="coerce")
rank["scalp_score"] = pd.to_numeric(rank.get("scalp_score", np.nan), errors="coerce")
rank["swing_score"] = pd.to_numeric(rank.get("swing_score", np.nan), errors="coerce")
rank["swing_ret10"] = pd.to_numeric(rank.get("swing_ret10", np.nan), errors="coerce")
rank["swing_ret20"] = pd.to_numeric(rank.get("swing_ret20", np.nan), errors="coerce")

# -----------------------------
# Ranking per mode
# -----------------------------
if rank_mode == "Scalp (Intraday Breakout)":
    rank = rank.sort_values(
        ["scalp_score", "scalp_early_ratio", "scalp_up_dom", "tiebreak"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    rank = rank[rank["scalp_score"] >= float(min_scalp)].copy()
    if rank.empty:
        st.warning("Semua kandidat ke-filter oleh min_scalp. Turunin min_scalp dulu.")
        st.stop()

elif rank_mode == "Swing (Multi-day Trend)":
    rank = rank.sort_values(
        ["swing_score", "swing_ret20", "swing_ret10", "rel_vol", "tiebreak"],
        ascending=[False, False, False, False, False]
    ).reset_index(drop=True)

    rank = rank[rank["swing_score"] >= float(min_swing)].copy()
    if rank.empty:
        st.warning("Semua kandidat ke-filter oleh min_swing. Turunin min_swing dulu.")
        st.stop()

else:
    rank = rank.sort_values(
        ["score", "S_activity", "S_vpa", "S_hap", "rel_vol", "tiebreak"],
        ascending=[False, False, False, False, False, False]
    ).reset_index(drop=True)

    rank = rank[rank["score"] >= float(min_score)].copy()
    if rank.empty:
        st.warning("Semua kandidat ke-filter oleh min_score. Turunin min_score dulu.")
        st.stop()

# -----------------------------
# Table
# -----------------------------
st.markdown("#### ✅ Top Candidates")

show_cols = [
    "ticker", "scan_date",
    "score", "S_activity", "S_vpa", "S_hap",
    "scalp_score", "scalp_early_ratio", "scalp_up_dom",
    "swing_score", "swing_ret10", "swing_ret20", "swing_close_pos_20", "swing_volr",
    "rel_vol", "ret1d", "ret_day", "close_pos", "early_vs_prev",
    "day_vol_5m_sum", "last_close"
]
show = rank[[c for c in show_cols if c in rank.columns]].copy()

st.dataframe(
    show.assign(
        score=show["score"].map(lambda x: f"{x:.1f}" if pd.notna(x) else ""),
        scalp_score=show["scalp_score"].map(lambda x: f"{x:.1f}" if pd.notna(x) else ""),
        scalp_early_ratio=show.get("scalp_early_ratio", pd.Series(dtype=float)).map(lambda x: f"{x:.2f}" if pd.notna(x) else ""),
        scalp_up_dom=show.get("scalp_up_dom", pd.Series(dtype=float)).map(lambda x: f"{x:.2f}" if pd.notna(x) else ""),
        rel_vol=show["rel_vol"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}"),
        ret1d=show["ret1d"].map(lambda x: "" if pd.isna(x) else f"{x*100:.2f}%"),
        ret_day=show["ret_day"].map(lambda x: f"{x*100:.2f}%"),
        close_pos=show["close_pos"].map(lambda x: f"{x:.2f}"),
        early_vs_prev=show["early_vs_prev"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}"),
        S_activity=show["S_activity"].map(lambda x: f"{x:.1f}"),
        S_vpa=show["S_vpa"].map(lambda x: f"{x:.1f}"),
        S_hap=show["S_hap"].map(lambda x: f"{x:.1f}"),
        day_vol_5m_sum=show["day_vol_5m_sum"].map(lambda x: f"{x:,.0f}"),
        last_close=show["last_close"].map(lambda x: f"{x:,.2f}"),
        swing_score=show.get("swing_score", pd.Series(dtype=float)).map(lambda x: f"{x:.1f}" if pd.notna(x) else ""),
        swing_ret10=show.get("swing_ret10", pd.Series(dtype=float)).map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else ""),
        swing_ret20=show.get("swing_ret20", pd.Series(dtype=float)).map(lambda x: f"{x*100:.2f}%" if pd.notna(x) else ""),
        swing_close_pos_20=show.get("swing_close_pos_20", pd.Series(dtype=float)).map(lambda x: f"{x:.2f}" if pd.notna(x) else ""),
        swing_volr=show.get("swing_volr", pd.Series(dtype=float)).map(lambda x: f"{x:.2f}" if pd.notna(x) else ""),
    ),
    use_container_width=True,
    hide_index=True
)

st.divider()

# -----------------------------
# Inspector
# -----------------------------
st.markdown("### 🔎 Inspect Stock (5m)")
pick = st.selectbox("Select stock", options=rank["ticker"].tolist(), index=0)

df5 = price_map_5m.get(pick, pd.DataFrame())
if df5.empty:
    st.warning("No 5m data for selected stock.")
    st.stop()

days = intraday_groups(df5)
scan_date = sorted(days.keys())[-1]
day_df = days[scan_date]

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=day_df.index,
    open=day_df["Open"], high=day_df["High"], low=day_df["Low"], close=day_df["Close"],
    name="5m"
))
fig.update_layout(
    title=f"📈 5m Price — {pick} ({scan_date})",
    height=520,
    margin=dict(l=0, r=0, t=50, b=0),
    xaxis_rangeslider_visible=False
)
st.plotly_chart(fig, use_container_width=True)

vfig = go.Figure()
vfig.add_trace(go.Bar(x=day_df.index, y=day_df["Volume"], name="Volume"))
vfig.update_layout(
    title="📊 5m Volume",
    height=240,
    margin=dict(l=0, r=0, t=50, b=0)
)
st.plotly_chart(vfig, use_container_width=True)

st.markdown("#### ⚡ Fast Breakout + volume burst)")
events = hap_store.get(pick, pd.DataFrame())
if events is None or events.empty:
    st.caption("Tidak ada event yang memenuhi kriteria hari ini (atau datanya kurang).")
else:
    st.dataframe(events, use_container_width=True)
