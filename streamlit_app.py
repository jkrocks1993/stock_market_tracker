import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import zipfile
import io
import re
import difflib

st.set_page_config(page_title="Indian Stock Data Downloader + Predictor", page_icon="📈", layout="wide")

st.title("📈 Indian Stock Data Downloader + Predictor")
st.markdown("Download data • Explainable predictions • Ticker search • Max history lookup")

# ===================== SESSION STATE =====================
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "RELIANCE.NS"

# ===================== STOCK DATABASE =====================
STOCK_NAMES = [
    "Reliance Industries", "Tata Consultancy Services", "Infosys", "HDFC Bank",
    "ICICI Bank", "State Bank of India", "Bharti Airtel", "ITC", "Hindustan Unilever",
    "Larsen & Toubro", "Axis Bank", "Kotak Mahindra Bank", "Bajaj Finance",
    "Maruti Suzuki", "Tata Motors", "Mahindra & Mahindra", "Sun Pharma",
    "Dr Reddy's Laboratories", "Cipla", "Wipro", "HCL Technologies",
    "Tech Mahindra", "Power Grid Corporation", "NTPC", "Coal India",
    "Oil and Natural Gas Corporation", "Indian Oil Corporation", "Bharat Petroleum",
    "Tata Steel", "JSW Steel", "Hindalco Industries", "Adani Ports",
    "Adani Enterprises", "UltraTech Cement", "Shree Cement", "Grasim Industries",
    "Asian Paints", "Pidilite Industries", "Britannia Industries", "Nestle India",
    "Dabur India", "Godrej Consumer Products", "Colgate Palmolive", "Emami",
    "Zomato", "Paytm", "Nykaa", "PolicyBazaar", "PB Fintech",
    "HDFC Life", "SBI Life", "ICICI Prudential Life", "Bajaj Finserv",
    "Shriram Finance", "Cholamandalam Investment", "Muthoot Finance",
    "Eicher Motors", "Hero MotoCorp", "Bajaj Auto", "TVS Motor",
    "Divi's Laboratories", "Biocon", "Lupin", "Aurobindo Pharma",
    "Glenmark Pharmaceuticals", "Torrent Pharmaceuticals"
]

TICKER_MAP = {
    "reliance industries": "RELIANCE.NS", "tata consultancy services": "TCS.NS",
    "infosys": "INFY.NS", "hdfc bank": "HDFCBANK.NS", "icici bank": "ICICIBANK.NS",
    "state bank of india": "SBIN.NS", "bharti airtel": "BHARTIARTL.NS", "itc": "ITC.NS",
    "hindustan unilever": "HINDUNILVR.NS", "larsen & toubro": "LT.NS",
    "axis bank": "AXISBANK.NS", "kotak mahindra bank": "KOTAKBANK.NS",
    "bajaj finance": "BAJFINANCE.NS", "maruti suzuki": "MARUTI.NS",
    "tata motors": "TATAMOTORS.NS", "mahindra & mahindra": "M&M.NS",
    "sun pharma": "SUNPHARMA.NS", "dr reddy's laboratories": "DRREDDY.NS",
    "cipla": "CIPLA.NS", "wipro": "WIPRO.NS", "hcl technologies": "HCLTECH.NS",
    "tech mahindra": "TECHM.NS", "power grid corporation": "POWERGRID.NS",
    "ntpc": "NTPC.NS", "coal india": "COALINDIA.NS", "oil and natural gas corporation": "ONGC.NS",
    "indian oil corporation": "IOC.NS", "bharat petroleum": "BPCL.NS",
    "tata steel": "TATASTEEL.NS", "jsw steel": "JSWSTEEL.NS",
    "hindalco industries": "HINDALCO.NS", "adani ports": "ADANIPORTS.NS",
    "adani enterprises": "ADANIENT.NS", "ultratech cement": "ULTRACEMCO.NS",
    "shree cement": "SHREECEM.NS", "grasim industries": "GRASIM.NS",
    "asian paints": "ASIANPAINT.NS", "pidilite industries": "PIDILITIND.NS",
    "britannia industries": "BRITANNIA.NS", "nestle india": "NESTLEIND.NS",
    "dabur india": "DABUR.NS", "godrej consumer products": "GODREJCP.NS",
    "colgate palmolive": "COLPAL.NS", "emami": "EMAMILTD.NS",
    "zomato": "ZOMATO.NS", "paytm": "PAYTM.NS", "nykaa": "NYKAA.NS",
    "policybazaar": "POLICYBZR.NS", "pb fintech": "PBFINTECH.NS",
    "hdfc life": "HDFCLIFE.NS", "sbi life": "SBILIFE.NS",
    "icici prudential life": "ICICIPRULI.NS", "bajaj finserv": "BAJAJFINSV.NS",
    "shriram finance": "SHRIRAMFIN.NS", "cholamandalam investment": "CHOLAFIN.NS",
    "muthoot finance": "MUTHOOTFIN.NS", "eicher motors": "EICHERMOT.NS",
    "hero motocorp": "HEROMOTOCO.NS", "bajaj auto": "BAJAJ-AUTO.NS",
    "tvs motor": "TVSMOTOR.NS", "divi's laboratories": "DIVISLAB.NS",
    "biocon": "BIOCON.NS", "lupin": "LUPIN.NS", "aurobindo pharma": "AUROPHARMA.NS",
    "glenmark pharmaceuticals": "GLENMARK.NS", "torrent pharmaceuticals": "TORNTPHARM.NS",
}

def find_closest_tickers(query: str, n=5):
    if not query or len(query) < 2:
        return []
    query_lower = query.lower().strip()
    matches = difflib.get_close_matches(query_lower, STOCK_NAMES, n=n, cutoff=0.45)
    return [(name, TICKER_MAP.get(name.lower())) for name in matches if TICKER_MAP.get(name.lower())]

# ===================== HELPER FUNCTIONS =====================
def fetch_stock_data(ticker, start=None, end=None, period=None, interval="1d"):
    try:
        if period:
            return yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        return yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
    except:
        return pd.DataFrame()

def parse_tickers(text):
    if not text: return []
    return [t.strip().upper() for t in re.split(r'[,\s\n]+', text.strip()) if t.strip()]

def predict_with_reasoning(df, horizon_days=5, lookback=60):
    if df.empty or len(df) < 10:
        return None, None, "Insufficient data", 0, 0
    close = df['Close'].dropna()
    if len(close) < lookback: lookback = max(10, len(close)//2)
    recent = close.iloc[-lookback:]
    last_price = close.iloc[-1]

    mom_period = min(30, len(close)-1)
    momentum = ((last_price / close.iloc[-mom_period-1]) - 1) * 100 if mom_period > 0 else 0

    x = np.arange(len(recent))
    slope, _ = np.polyfit(x, recent.values, 1)
    r2 = np.corrcoef(x, recent.values)[0,1]**2 if len(x) > 1 else 0

    pred_trend = last_price + slope * horizon_days
    rets = recent.pct_change().dropna()
    avg_ret = rets.mean() if len(rets) > 0 else 0
    pred_mom = last_price * (1 + avg_ret * horizon_days)

    pred_price = (pred_trend + pred_mom) / 2
    pct = ((pred_price / last_price) - 1) * 100
    direction = "UP" if pct > 0 else "DOWN"
    vol = rets.std() * 100 if len(rets) > 0 else 0

    reasoning = f"""**Model Reasoning ({horizon_days} days)**
- **Momentum**: {momentum:.2f}% last {mom_period} days → **{'Bullish' if momentum > 1 else 'Bearish' if momentum < -1 else 'Neutral'}**
- **Linear Trend**: ₹{slope:.2f}/day | R² = {r2:.2f}
- **Volatility**: {vol:.2f}%
- **Conclusion**: Price expected to move **{direction}** by ~**{pct:.2f}%**"""
    return pred_price, pct, reasoning, slope, momentum

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("⚙️ Settings & Tools")
    today = datetime.now().date()
    start_date = st.date_input("Start Date", value=today - timedelta(days=365*3))
    end_date = st.date_input("End Date", value=today)
    interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

    st.divider()
    st.subheader("🔎 Find Ticker by Company Name")
    name_query = st.text_input("Type company name (e.g. Reliance, HDFC, TCS, Zomato)")

    if name_query:
        matches = find_closest_tickers(name_query)
        if matches:
            st.write("**Click to auto-fill:**")
            for name, ticker in matches:
                if st.button(f"📋 {name} → {ticker}", key=f"fill_{ticker}"):
                    st.session_state.selected_ticker = ticker
                    st.success(f"✅ Auto-filled: {ticker}")
                    st.rerun()
        else:
            st.info("No close matches. Try a different spelling.")

# ===================== TABS =====================
tab1, tab2, tab3 = st.tabs(["🔹 Single Stock", "🔹 Batch Download", "🔮 Prediction + Reasoning"])

# ===================== TAB 1: SINGLE =====================
with tab1:
    st.subheader("Download Data for One Stock")
    ticker = st.text_input(
        "Ticker Symbol",
        value=st.session_state.selected_ticker,
        key="single_ticker_input"
    ).strip().upper()

    if st.button("🚀 Fetch Data", type="primary", use_container_width=True):
        if ticker:
            with st.spinner(f"Downloading {ticker}..."):
                data = fetch_stock_data(ticker, start_date, end_date, interval=interval)
                if data.empty:
                    st.error("No data found for this ticker.")
                else:
                    st.success(f"✅ {len(data)} rows fetched")
                    st.dataframe(data, use_container_width=True, height=420)
                    csv = data.to_csv().encode("utf-8")
                    st.download_button("⬇️ Download CSV", csv, f"{ticker}_{start_date}_{end_date}.csv", "text/csv")

# ===================== TAB 2: BATCH =====================
with tab2:
    st.subheader("Batch Download Multiple Stocks")
    batch_text = st.text_area("Tickers (one per line or comma separated)", height=110)
    batch_file = st.file_uploader("Or upload .txt/.csv file", type=["txt", "csv"])

    if st.button("📥 Download Batch Data", type="primary"):
        tickers = parse_tickers(batch_text)
        if batch_file:
            tickers += parse_tickers(batch_file.read().decode("utf-8"))
        tickers = sorted(list(set(tickers)))

        if not tickers:
            st.warning("Please enter at least one ticker.")
        else:
            progress = st.progress(0)
            all_data = []
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, t in enumerate(tickers):
                    progress.progress((i + 1) / len(tickers), text=f"Fetching {t}...")
                    df = fetch_stock_data(t, start_date, end_date, interval=interval)
                    if not df.empty:
                        all_data.append(df.assign(Ticker=t))
                        zf.writestr(f"{t}.csv", df.to_csv())
            progress.empty()
            if all_data:
                combined = pd.concat(all_data).reset_index()
                st.success(f"✅ Downloaded {len(all_data)} stocks successfully!")
                st.dataframe(combined.head(500), use_container_width=True, height=350)
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("⬇️ Combined CSV (All Tickers)", combined.to_csv(index=False).encode(), "combined_stocks.csv")
                with c2:
                    zip_buffer.seek(0)
                    st.download_button("⬇️ Download ZIP (Individual CSVs)", zip_buffer.getvalue(), "stock_data_batch.zip", "application/zip")

# ===================== TAB 3: PREDICTION =====================
with tab3:
    st.subheader("🔮 Multi-Stock Price Prediction with Reasoning")

    # Quick use of selected ticker
    if st.session_state.selected_ticker:
        if st.button(f"➕ Add selected ticker ({st.session_state.selected_ticker}) to prediction list"):
            st.session_state.pred_tickers = st.session_state.get("pred_tickers", "") + f"\n{st.session_state.selected_ticker}"
            st.rerun()

    pred_text = st.text_area(
        "Enter tickers for prediction",
        value=st.session_state.get("pred_tickers", ""),
        height=90,
        placeholder="RELIANCE.NS\nTCS.NS\nINFY.NS"
    )

    horizon = st.slider("Forecast Horizon (days)", 1, 14, 5)

    col_run, col_history = st.columns([2, 1])
    with col_run:
        if st.button("🔮 Run Prediction & Reasoning", type="primary", use_container_width=True):
            tickers = parse_tickers(pred_text)
            if not tickers:
                st.warning("Please enter at least one ticker.")
            else:
                results = []
                progress = st.progress(0)
                for i, ticker in enumerate(tickers):
                    progress.progress((i+1)/len(tickers), text=f"Analyzing {ticker}...")
                    df = fetch_stock_data(ticker, start_date, end_date, interval="1d")
                    if df.empty or len(df) < 20:
                        results.append({"Ticker": ticker, "Status": "Insufficient data"})
                        continue
                    pred_price, pct, reasoning, slope, mom = predict_with_reasoning(df, horizon)
                    current = df['Close'].iloc[-1]
                    results.append({
                        "Ticker": ticker,
                        "Current Price": round(current, 2),
                        "Predicted Price": round(pred_price, 2),
                        "Expected Change %": round(pct, 2),
                        "Momentum (30d)": round(mom, 2),
                        "Trend Slope": round(slope, 2),
                        "Reasoning": reasoning,
                        "Data": df
                    })
                progress.empty()

                # Summary Table
                st.subheader("📊 Prediction Summary")
                summary_df = pd.DataFrame([{k:v for k,v in r.items() if k not in ["Reasoning","Data"]} for r in results])
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                # Detailed Reasoning + Charts
                st.subheader("📝 Detailed Reasoning per Stock")
                for r in results:
                    if r.get("Status"):
                        with st.expander(f"❌ {r['Ticker']} — Insufficient data"): 
                            st.warning("Not enough historical data for reliable prediction.")
                        continue
                    with st.expander(f"📈 {r['Ticker']} | ₹{r['Current Price']} → ₹{r['Predicted Price']} ({r['Expected Change %']}%)"):
                        st.markdown(r["Reasoning"])
                        hist = r["Data"][['Close']].iloc[-60:]
                        future_dates = pd.date_range(start=hist.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="B")
                        future_prices = np.linspace(hist['Close'].iloc[-1], r['Predicted Price'], horizon)
                        future_df = pd.DataFrame({"Close": future_prices}, index=future_dates)
                        chart_df = pd.concat([hist['Close'], future_df['Close']])
                        st.line_chart(chart_df, use_container_width=True)

    with col_history:
        if st.button("🔍 Check Max Available History", use_container_width=True):
            tickers = parse_tickers(pred_text)
            if tickers:
                ticker = tickers[0]
                with st.spinner(f"Checking maximum history for {ticker}..."):
                    max_df = fetch_stock_data(ticker, period="max")
                    if not max_df.empty:
                        earliest = max_df.index.min().date()
                        years = (datetime.now().date() - earliest).days // 365
                        st.success(f"**{ticker}** has data since **{earliest}** ({years} years)")
                        st.caption(f"Total available rows: {len(max_df)}")
                    else:
                        st.error("No data found.")

st.markdown("---")
st.caption("Data from Yahoo Finance via yfinance | For educational & research use only. Predictions are not financial advice.")