import streamlit as st
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────
st.set_page_config(page_title="StockSignal AI", page_icon="📈", layout="wide")

API_URL = "http://localhost:8000"
SYMBOLS = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL", "META"]

BACKTEST_ACCURACY = {
    "AAPL": 61.2, "TSLA": 58.4, "NVDA": 64.7,
    "MSFT": 62.1, "AMZN": 60.8, "GOOGL": 63.3, "META": 61.9
}

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #050a0e;
    color: #c8d6df;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1500px; }

.terminal-header { border-bottom: 1px solid #0d2233; padding-bottom: 1.5rem; margin-bottom: 2rem; }
.terminal-title  { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 700; color: #00d4ff; letter-spacing: -0.5px; margin: 0; }
.terminal-sub    { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #2a4a5e; letter-spacing: 0.2em; text-transform: uppercase; margin-top: 0.25rem; }

.stTabs [data-baseweb="tab-list"] { gap: 0; background: transparent; border-bottom: 1px solid #0d2233; }
.stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.15em; color: #2a4a5e; padding: 0.6rem 1.5rem; background: transparent; border: none; text-transform: uppercase; }
.stTabs [aria-selected="true"] { color: #00d4ff; border-bottom: 2px solid #00d4ff; background: transparent; }

.signal-card { background: linear-gradient(135deg, #050f17 0%, #0a1a24 100%); border: 1px solid #0d2233; border-radius: 4px; padding: 2rem; position: relative; overflow: hidden; }
.signal-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #00d4ff, transparent); }
.signal-direction { font-family: 'IBM Plex Mono', monospace; font-size: 3.5rem; font-weight: 700; line-height: 1; margin-bottom: 0.5rem; }
.signal-up   { color: #00ff88; }
.signal-down { color: #ff4466; }
.signal-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.25em; color: #2a4a5e; text-transform: uppercase; }

.stat-box { background: #050f17; border: 1px solid #0d2233; border-radius: 4px; padding: 1.25rem; }
.stat-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 600; color: #00d4ff; line-height: 1; }
.stat-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; color: #2a4a5e; letter-spacing: 0.2em; text-transform: uppercase; margin-top: 0.4rem; }

.conf-track { background: #0d2233; height: 4px; border-radius: 2px; margin-top: 1rem; overflow: hidden; }
.conf-fill-up   { height: 100%; background: #00ff88; border-radius: 2px; }
.conf-fill-down { height: 100%; background: #ff4466; border-radius: 2px; }

.signal-text { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #4a7a8a; background: #050f17; border: 1px solid #0d2233; border-left: 3px solid #00d4ff; padding: 0.75rem 1rem; margin-top: 1rem; border-radius: 2px; }
.error-box   { background: #120508; border: 1px solid #3a0d15; border-left: 3px solid #ff4466; padding: 0.75rem 1rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #ff4466; border-radius: 2px; margin-top: 1rem; }
.section-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; letter-spacing: 0.2em; color: #2a4a5e; text-transform: uppercase; margin-bottom: 0.5rem; margin-top: 1.5rem; }

/* Scan row card */
.scan-row { display: flex; align-items: center; justify-content: space-between; padding: 0.75rem 1rem; border: 1px solid #0d2233; border-radius: 4px; margin-bottom: 0.5rem; background: #050f17; }
.scan-ticker { font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; font-weight: 600; color: #c8d6df; }
.scan-up   { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; font-weight: 600; color: #00ff88; }
.scan-down { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; font-weight: 600; color: #ff4466; }
.scan-conf { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #4a7a8a; }
.scan-sent { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #2a4a5e; }

.stSelectbox label { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.2em; color: #2a4a5e; text-transform: uppercase; }
.stButton > button { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.15em; text-transform: uppercase; background: transparent; border: 1px solid #00d4ff; color: #00d4ff; padding: 0.6rem 2rem; border-radius: 2px; width: 100%; }
.stButton > button:hover { background: #00d4ff; color: #050a0e; }
.stSpinner > div { border-top-color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────

def get_signal_text(direction, confidence):
    if direction == "up" and confidence > 0.70:
        return "Strong bullish signal — technical momentum confirmed"
    elif direction == "up":
        return "Moderate bullish signal — proceed with caution"
    elif direction == "down" and confidence < 0.30:
        return "Strong bearish signal — technicals override sentiment"
    else:
        return "Moderate bearish signal — mixed indicators"

def format_confidence(direction, raw_conf):
    return round(raw_conf * 100 if direction == "up" else (1 - raw_conf) * 100, 1)

def fetch_price_chart(ticker):
    df = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df["MA7"]  = df["Close"].rolling(7).mean()
    df["MA30"] = df["Close"].rolling(30).mean()
    return df.dropna()

def render_price_chart(ticker):
    with st.spinner(f"Loading {ticker} price data..."):
        df = fetch_price_chart(ticker)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        increasing_line_color="#00ff88",
        decreasing_line_color="#ff4466",
        increasing_fillcolor="#00ff88",
        decreasing_fillcolor="#ff4466",
        name="Price",
        showlegend=False
    ), row=1, col=1)

    # Moving averages
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA7"],
        line=dict(color="#00d4ff", width=1.5),
        name="MA7"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA30"],
        line=dict(color="#f59e0b", width=1.5, dash="dot"),
        name="MA30"
    ), row=1, col=1)

    # Volume
    colors = ["#00ff88" if c >= o else "#ff4466"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=colors, opacity=0.6,
        name="Volume", showlegend=False
    ), row=2, col=1)

    fig.update_layout(
        plot_bgcolor="#050a0e", paper_bgcolor="#050a0e",
        font=dict(family="IBM Plex Mono", color="#4a7a8a", size=10),
        xaxis=dict(gridcolor="#0d2233", linecolor="#0d2233",
                   rangeslider=dict(visible=False)),
        xaxis2=dict(gridcolor="#0d2233", linecolor="#0d2233"),
        yaxis=dict(gridcolor="#0d2233", linecolor="#0d2233"),
        yaxis2=dict(gridcolor="#0d2233", linecolor="#0d2233"),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    orientation="h", y=1.02, x=0),
        margin=dict(l=0, r=0, t=30, b=0),
        height=380
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class="terminal-header">
    <div class="terminal-title">▸ STOCKSIGNAL AI</div>
    <div class="terminal-sub">LSTM · FinBERT · 7 Equities · 62.07% Directional Accuracy</div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["SIGNAL", "SCAN ALL", "HISTORY", "BACKTEST"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — SIGNAL
# ══════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        ticker = st.selectbox("SELECT TICKER", options=SYMBOLS, index=0)
        run    = st.button("RUN ANALYSIS", use_container_width=True)

        st.markdown("""
        <div style="margin-top:1.5rem">
            <div class="section-label">MODEL INFO</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#4a7a8a; line-height:2;">
                Architecture: 2-layer LSTM<br>
                Features: 12 technical indicators<br>
                Sentiment: FinBERT (ProsusAI)<br>
                Training: 7 stocks · 5 years<br>
                Test accuracy: 62.07%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        if run:
            with st.spinner(f"Fetching {ticker} data and running model..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/predict",
                        json={"ticker": ticker},
                        timeout=90
                    )
                    if resp.status_code == 200:
                        r          = resp.json()
                        direction  = r["direction"]
                        confidence = r["confidence"]
                        sentiment  = r["sentiment"]
                        timestamp  = r["timestamp"]
                        conf_pct   = format_confidence(direction, confidence)
                        bar_width  = int(conf_pct)
                        dir_class  = "signal-up" if direction == "up" else "signal-down"
                        dir_symbol = "▲ UP" if direction == "up" else "▼ DOWN"
                        fill_class = "conf-fill-up" if direction == "up" else "conf-fill-down"
                        sig_text   = get_signal_text(direction, confidence)
                        sent_color = "#00ff88" if sentiment == "positive" else \
                                     "#ff4466" if sentiment == "negative" else "#00d4ff"

                        st.markdown(f"""
                        <div class="signal-card">
                            <div class="signal-label">{ticker} · 5-DAY DIRECTION FORECAST</div>
                            <div class="signal-direction {dir_class}">{dir_symbol}</div>
                            <div class="conf-track">
                                <div class="{fill_class}" style="width:{bar_width}%"></div>
                            </div>
                            <div class="signal-text">⟩ {sig_text}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown(f"""<div class="stat-box">
                                <div class="stat-value">{conf_pct}%</div>
                                <div class="stat-label">Confidence</div>
                            </div>""", unsafe_allow_html=True)
                        with c2:
                            st.markdown(f"""<div class="stat-box">
                                <div class="stat-value" style="color:{sent_color};font-size:1rem">
                                    {sentiment.upper()}
                                </div>
                                <div class="stat-label">News Sentiment</div>
                            </div>""", unsafe_allow_html=True)
                        with c3:
                            st.markdown(f"""<div class="stat-box">
                                <div class="stat-value" style="font-size:0.8rem;color:#4a7a8a">
                                    {timestamp}
                                </div>
                                <div class="stat-label">Timestamp</div>
                            </div>""", unsafe_allow_html=True)

                    else:
                        st.markdown(f'<div class="error-box">⚠ API error {resp.status_code}</div>',
                                    unsafe_allow_html=True)

                except requests.exceptions.ConnectionError:
                    st.markdown('<div class="error-box">⚠ Cannot connect to API on port 8000</div>',
                                unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="error-box">⚠ {str(e)}</div>',
                                unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;
                        height:220px;border:1px dashed #0d2233;border-radius:4px;">
                <div style="text-align:center">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:2rem;color:#0d2233;margin-bottom:0.5rem;">◈</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;letter-spacing:0.2em;color:#1a3344;text-transform:uppercase;">
                        Select ticker and run analysis
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Price chart always visible
        st.markdown('<div class="section-label">PRICE CHART — LAST 3 MONTHS</div>',
                    unsafe_allow_html=True)
        render_price_chart(ticker)

# ══════════════════════════════════════════════════════════════
# TAB 2 — SCAN ALL
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#4a7a8a;margin-bottom:1.5rem;">
        Run the model on all 7 tickers simultaneously and compare signals.
    </div>
    """, unsafe_allow_html=True)

    scan_all = st.button("⟩ SCAN ALL TICKERS", use_container_width=False)

    if scan_all:
        results = []
        progress = st.progress(0)
        status   = st.empty()

        for i, sym in enumerate(SYMBOLS):
            status.markdown(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:#4a7a8a;">Analyzing {sym}...</div>',
                unsafe_allow_html=True
            )
            try:
                resp = requests.post(f"{API_URL}/predict", json={"ticker": sym}, timeout=90)
                if resp.status_code == 200:
                    r = resp.json()
                    results.append(r)
            except:
                pass
            progress.progress((i + 1) / len(SYMBOLS))

        status.empty()
        progress.empty()

        if results:
            # Summary stats
            up_count   = sum(1 for r in results if r["direction"] == "up")
            down_count = len(results) - up_count

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="stat-box">
                    <div class="stat-value">{len(results)}</div>
                    <div class="stat-label">Tickers Scanned</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="stat-box">
                    <div class="stat-value" style="color:#00ff88">{up_count} ▲</div>
                    <div class="stat-label">Bullish Signals</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="stat-box">
                    <div class="stat-value" style="color:#ff4466">{down_count} ▼</div>
                    <div class="stat-label">Bearish Signals</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">SIGNAL SUMMARY</div>', unsafe_allow_html=True)

            # Sort by confidence strength
            results.sort(key=lambda r: abs(r["confidence"] - 0.5), reverse=True)

            for r in results:
                direction  = r["direction"]
                confidence = r["confidence"]
                conf_pct   = format_confidence(direction, confidence)
                dir_sym    = "▲ UP" if direction == "up" else "▼ DOWN"
                dir_cls    = "scan-up" if direction == "up" else "scan-down"
                sent_color = "#00ff88" if r["sentiment"] == "positive" else \
                             "#ff4466" if r["sentiment"] == "negative" else "#4a7a8a"
                bar_w      = int(conf_pct)
                bar_col    = "#00ff88" if direction == "up" else "#ff4466"

                st.markdown(f"""
                <div class="scan-row">
                    <div style="display:flex;align-items:center;gap:2rem;flex:1">
                        <div class="scan-ticker">{r["ticker"]}</div>
                        <div class="{dir_cls}">{dir_sym}</div>
                        <div style="flex:1;max-width:200px">
                            <div style="background:#0d2233;height:3px;border-radius:2px;overflow:hidden">
                                <div style="width:{bar_w}%;height:100%;background:{bar_col};border-radius:2px"></div>
                            </div>
                        </div>
                        <div class="scan-conf">{conf_pct}% confidence</div>
                    </div>
                    <div class="scan-sent" style="color:{sent_color}">{r["sentiment"].upper()}</div>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:center;
                    height:200px;border:1px dashed #0d2233;border-radius:4px;margin-top:1rem;">
            <div style="text-align:center">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                            letter-spacing:0.2em;color:#1a3344;text-transform:uppercase;">
                    Press scan to analyze all tickers
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ══════════════════════════════════════════════════════════════
with tab3:
    try:
        resp = requests.get(f"{API_URL}/predictions", timeout=10)
        if resp.status_code == 200:
            predictions = resp.json()["predictions"]
            if predictions:
                df = pd.DataFrame(predictions,
                                  columns=["ID", "Ticker", "Direction",
                                           "Confidence", "Sentiment", "Timestamp"])

                total      = len(df)
                up_count   = len(df[df["Direction"] == "up"])
                down_count = len(df[df["Direction"] == "down"])
                avg_conf   = round(df["Confidence"].apply(
                    lambda x: x if x > 0.5 else 1 - x).mean() * 100, 1)

                c1, c2, c3, c4 = st.columns(4)
                for col, val, label, color in [
                    (c1, total,      "Total Signals",   "#00d4ff"),
                    (c2, f"{up_count} ▲", "Bullish", "#00ff88"),
                    (c3, f"{down_count} ▼", "Bearish", "#ff4466"),
                    (c4, f"{avg_conf}%", "Avg Confidence", "#00d4ff")
                ]:
                    with col:
                        st.markdown(f"""<div class="stat-box">
                            <div class="stat-value" style="color:{color}">{val}</div>
                            <div class="stat-label">{label}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

                # Filter
                tickers_all = ["ALL"] + sorted(df["Ticker"].unique().tolist())
                fcol, _ = st.columns([1, 3])
                with fcol:
                    selected = st.selectbox("FILTER BY TICKER", tickers_all)

                df_show = df if selected == "ALL" else df[df["Ticker"] == selected]
                df_show = df_show.copy()
                df_show["Confidence"] = df_show["Confidence"].apply(
                    lambda x: f"{format_confidence('up' if x > 0.5 else 'down', x)}%"
                )
                df_show["Direction"] = df_show["Direction"].str.upper()
                df_show["Sentiment"] = df_show["Sentiment"].str.upper()
                df_show = df_show.drop("ID", axis=1)

                st.markdown('<div class="section-label">ALL PREDICTIONS</div>', unsafe_allow_html=True)
                st.dataframe(df_show, use_container_width=True, hide_index=True)

                # Distribution chart
                if len(df) > 1:
                    st.markdown('<div class="section-label">SIGNAL DISTRIBUTION BY TICKER</div>',
                                unsafe_allow_html=True)
                    counts = df.groupby(["Ticker", "Direction"]).size().reset_index(name="Count")
                    fig = go.Figure()
                    for direction, color in [("up", "#00ff88"), ("down", "#ff4466")]:
                        sub = counts[counts["Direction"] == direction]
                        fig.add_trace(go.Bar(
                            x=sub["Ticker"], y=sub["Count"],
                            name=direction.upper(),
                            marker_color=color, marker_line_width=0, opacity=0.85
                        ))
                    fig.update_layout(
                        barmode="group", plot_bgcolor="#050a0e", paper_bgcolor="#050a0e",
                        font=dict(family="IBM Plex Mono", color="#4a7a8a", size=10),
                        xaxis=dict(gridcolor="#0d2233", linecolor="#0d2233"),
                        yaxis=dict(gridcolor="#0d2233", linecolor="#0d2233"),
                        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
                        margin=dict(l=0, r=0, t=10, b=0), height=260
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("""
                <div style="display:flex;align-items:center;justify-content:center;
                            height:200px;border:1px dashed #0d2233;border-radius:4px;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;
                                letter-spacing:0.2em;color:#1a3344;text-transform:uppercase;">
                        No signals recorded yet
                    </div>
                </div>""", unsafe_allow_html=True)
    except requests.exceptions.ConnectionError:
        st.markdown('<div class="error-box">⚠ Cannot connect to API on port 8000</div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — BACKTEST
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;color:#4a7a8a;margin-bottom:1.5rem;line-height:1.8;">
        Directional accuracy measured on held-out test set (most recent 20% of data per ticker).<br>
        Random baseline = 50%. Academic benchmark = 52–56%.
    </div>
    """, unsafe_allow_html=True)

    # Overall accuracy stat
    overall = round(np.mean(list(BACKTEST_ACCURACY.values())), 2)
    best_t  = max(BACKTEST_ACCURACY, key=BACKTEST_ACCURACY.get)
    worst_t = min(BACKTEST_ACCURACY, key=BACKTEST_ACCURACY.get)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-value">{overall}%</div>
            <div class="stat-label">Overall Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-value" style="color:#00ff88">{best_t} {BACKTEST_ACCURACY[best_t]}%</div>
            <div class="stat-label">Best Ticker</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-value" style="color:#ff4466">{worst_t} {BACKTEST_ACCURACY[worst_t]}%</div>
            <div class="stat-label">Weakest Ticker</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">ACCURACY BY TICKER</div>', unsafe_allow_html=True)

    tickers = list(BACKTEST_ACCURACY.keys())
    accs    = list(BACKTEST_ACCURACY.values())
    colors  = ["#00ff88" if a >= 62 else "#00d4ff" if a >= 60 else "#f59e0b"
               for a in accs]

    fig = go.Figure()
    # Baseline reference line
    fig.add_hline(y=50, line_dash="dot", line_color="#2a4a5e",
                  annotation_text="Random baseline 50%",
                  annotation_font=dict(family="IBM Plex Mono", size=10, color="#2a4a5e"))
    fig.add_hline(y=56, line_dash="dot", line_color="#f59e0b",
                  annotation_text="Academic benchmark 56%",
                  annotation_font=dict(family="IBM Plex Mono", size=10, color="#f59e0b"))

    fig.add_trace(go.Bar(
        x=tickers, y=accs,
        marker_color=colors,
        marker_line_width=0,
        text=[f"{a}%" for a in accs],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11, color="#c8d6df"),
        showlegend=False
    ))

    fig.update_layout(
        plot_bgcolor="#050a0e", paper_bgcolor="#050a0e",
        font=dict(family="IBM Plex Mono", color="#4a7a8a", size=11),
        xaxis=dict(gridcolor="#0d2233", linecolor="#0d2233"),
        yaxis=dict(gridcolor="#0d2233", linecolor="#0d2233",
                   range=[45, 70], ticksuffix="%"),
        margin=dict(l=0, r=0, t=20, b=0),
        height=340
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#2a4a5e;
                line-height:1.8;margin-top:1rem;border-top:1px solid #0d2233;padding-top:1rem;">
        KEY FINDING: For high-volatility stocks like TSLA, the model learned that technical 
        indicators carry more predictive weight than news sentiment — consistent with the 
        "buy the rumor, sell the news" phenomenon observed in financial markets.
    </div>
    """, unsafe_allow_html=True)