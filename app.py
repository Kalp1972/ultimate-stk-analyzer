# app.py (FIXED VERSION)
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats  # For linregress (replaces OLS)
import matplotlib.pyplot as plt
from datetime import datetime
import io

st.set_page_config(page_title="Ultimate Indian Quant Analyzer (Fixed)", layout="wide")
st.title("🔧 Fixed Ultimate Indian Quant Analyzer")
st.markdown("**Stat Arb + Supertrend + RSI(2) + VWAP + Volume Profile** *(No statsmodels — Pure NumPy/SciPy)*")

# === CONFIG ===
default_config = {
    'capital': 500000,
    'risk_pct': 0.02,
    'entry_z': 2.0,
    'rsi': {'period': 2, 'oversold': 10, 'overbought': 90},
    'supertrend': {'period': 7, 'multiplier': 2.0},
    'volume_profile': {'bins': 50, 'value_area_pct': 0.7},
    'lots': {'nifty': 25, 'banknifty': 15},
    'futures': {'nifty_price': 24850, 'banknifty_price': 51200,
                'nifty_symbol': 'NIFTY25N27FUT', 'banknifty_symbol': 'BANKNIFTY25N27FUT'}
}

config = st.sidebar.expander("⚙️ Settings", expanded=True)
with config:
    capital = st.number_input("Capital (₹)", value=500000)
    risk_pct = st.slider("Risk %", 0.5, 5.0, 2.0) / 100
    entry_z = st.slider("Entry z-score", 1.5, 3.0, 2.0)
    st.markdown("**Futures Prices**")
    n_price = st.number_input("NIFTY FUT Price", value=24850)
    b_price = st.number_input("BANKNIFTY FUT Price", value=51200)

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("📁 Upload CSV (Date, Time, NIFTY_*, BANKNIFTY_*)", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        required = [f'{s}_{t}' for s in ['NIFTY', 'BANKNIFTY'] for t in ['Open','High','Low','Close','Volume']]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"❌ Missing columns: {missing[:5]}...")
            st.stop()

        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.set_index('DateTime').drop(['Date', 'Time'], axis=1).dropna()

        # === INDICATORS (NO STATS MODELS) ===
        # RSI(2)
        delta = df['NIFTY_Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(2).mean()
        loss = -delta.where(delta < 0, 0).rolling(2).mean()
        rs = gain / loss.replace(0, np.nan)  # Avoid div by zero
        df['RSI_2'] = 100 - (100 / (1 + rs))

        # VWAP (Daily Reset)
        df['TPV'] = df['NIFTY_Close'] * df['NIFTY_Volume']
        df['Date'] = df.index.date
        df['CumTPV'] = df.groupby('Date')['TPV'].cumsum()
        df['CumVol'] = df.groupby('Date')['NIFTY_Volume'].cumsum()
        df['VWAP'] = df['CumTPV'] / df['CumVol']

        # Supertrend
        high, low, close = df['NIFTY_High'], df['NIFTY_Low'], df['NIFTY_Close']
        tr = pd.DataFrame(index=df.index)
        tr['tr0'] = abs(high - low)
        tr['tr1'] = abs(high - close.shift())
        tr['tr2'] = abs(low - close.shift())
        atr = tr[['tr0', 'tr1', 'tr2']].max(axis=1).rolling(7).mean()
        hl2 = (high + low) / 2
        upper = hl2 + 2.0 * atr
        lower = hl2 - 2.0 * atr
        st = pd.Series(index=df.index, dtype=float)
        trend = pd.Series(index=df.index, dtype=int)
        for i in range(7, len(df)):
            curr, prev = i, i - 1
            if pd.isna(st.iloc[prev]):
                st.iloc[curr] = lower.iloc[curr]
                trend.iloc[curr] = 1
            elif close.iloc[curr] > upper.iloc[prev]:
                st.iloc[curr] = max(lower.iloc[curr], st.iloc[prev])
                trend.iloc[curr] = 1
            else:
                st.iloc[curr] = min(upper.iloc[curr], st.iloc[prev])
                trend.iloc[curr] = -1
        df['Supertrend'] = st
        df['Trend'] = trend

        # Volume Profile (Simplified POC/VAH/VAL)
        prices = df['NIFTY_Close'].values
        volumes = df['NIFTY_Volume'].values
        hist, edges = np.histogram(prices, bins=50, weights=volumes)
        poc_idx = np.argmax(hist)
        poc = (edges[poc_idx] + edges[poc_idx + 1]) / 2
        total_vol = volumes.sum()
        cum_vol = 0
        vah, val = poc, poc
        sorted_idx = np.argsort(hist)[::-1]
        for idx in sorted_idx:
            if cum_vol >= total_vol * 0.7:
                break
            left, right = edges[idx], edges[idx + 1]
            vol = hist[idx]
            cum_vol += vol
            vah = max(vah, right)
            val = min(val, left)

        # Stat Arb (Pure SciPy: Linregress for beta, Simple mean/std for z)
        y = df['NIFTY_Close'].values
        x = df['BANKNIFTY_Close'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        beta = slope  # Hedge ratio
        spread = y - beta * x
        mu = np.mean(spread)
        sigma = np.std(spread)
        z = (spread[-1] - mu) / sigma  # Simple z-score

        # Signal Logic
        stat_signal = 1 if z < -entry_z else -1 if z > entry_z else 0
        rsi = df['RSI_2'].iloc[-1]
        price = df['NIFTY_Close'].iloc[-1]
        vwap = df['VWAP'].iloc[-1]
        trend = df['Trend'].iloc[-1]

        final_signal = 0
        reason = "No z-signal"
        if stat_signal == 1 and not pd.isna(rsi) and rsi < 10 and price < vwap and trend == 1:
            final_signal = 1
            reason = "ALL FILTERS PASS ✅"
        elif stat_signal == -1 and not pd.isna(rsi) and rsi > 90 and price > vwap and trend == -1:
            final_signal = -1
            reason = "ALL FILTERS PASS ✅"
        else:
            reasons = []
            if stat_signal == 0:
                reasons.append("z-score weak")
            if 'rsi' in locals() and stat_signal == 1 and rsi >= 10:
                reasons.append("RSI not oversold")
            if 'rsi' in locals() and stat_signal == -1 and rsi <= 90:
                reasons.append("RSI not overbought")
            if stat_signal == 1 and price >= vwap:
                reasons.append("Price ≥ VWAP")
            if stat_signal == -1 and price <= vwap:
                reasons.append("Price ≤ VWAP")
            if stat_signal == 1 and trend != 1:
                reasons.append("Supertrend not UP")
            if stat_signal == -1 and trend != -1:
                reasons.append("Supertrend not DOWN")
            reason = " | ".join(reasons)

        # Trade Suggestion
        if final_signal != 0:
            risk = capital * risk_pct
            contracts = max(1, int(risk / (abs(z) * 100 * 25 * 0.5)))
            n_qty = contracts * 25
            b_qty = int(contracts * 15 * beta)
            side = "🟢 BUY" if final_signal == 1 else "🔴 SELL"
            opp = "🔴 SELL" if final_signal == 1 else "🟢 BUY"
            target = f"Target: VAH {'↑' if final_signal==1 else '↓'} ₹{vah:.0f}"
            stop = f"Stop: POC ₹{poc:.0f}"
            trade = f"{side} {n_qty} {default_config['futures']['nifty_symbol']} @ ~₹{n_price:,}\n{opp} {b_qty} {default_config['futures']['banknifty_symbol']} @ ~₹{b_price:,}\nRisk: ₹{int(risk):,} | {target} | {stop}"
        else:
            trade = "⏸️ HOLD / EXIT — No strong signal"

        # === DASHBOARD ===
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("📊 z-score", f"{z:+.2f}")
            st.metric("📈 RSI(2)", f"{rsi:.1f}" if not pd.isna(rsi) else "N/A")
            st.metric("📍 VWAP", f"₹{vwap:.0f:,}")
            st.metric("🎯 Trend", "🟢 UP" if trend == 1 else "🔴 DOWN")
            st.metric("🔍 POC", f"₹{poc:.0f:,}")
            st.markdown(f"### **Signal: {reason}**")
            st.code(trade, language="text")

        with col2:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
            
            # Spread Plot
            ax1.plot(spread, label='Spread', color='purple', linewidth=1)
            ax1.axhline(mu, color='green', linestyle='--', label=f'Mean: ₹{mu:.0f}')
            ax1.axhspan(mu - 2*sigma, mu + 2*sigma, alpha=0.2, color='gray', label='±2σ')
            ax1.set_title(f"Stat Arb Spread | z = {z:+.2f}")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Price + Indicators
            ax2.plot(df['NIFTY_Close'], label='NIFTY Close', color='black', alpha=0.7)
            ax2.plot(df['VWAP'], label='VWAP', color='orange', linewidth=2)
            ax2.plot(df['Supertrend'], label='Supertrend', color='green' if trend==1 else 'red', linewidth=2)
            ax2.set_title("NIFTY + VWAP + Supertrend")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Volume Profile
            centers = (edges[:-1] + edges[1:]) / 2
            ax3.barh(centers, hist, height=(edges[1] - edges[0]), alpha=0.7, color='blue')
            ax3.axhline(poc, color='red', linestyle='--', linewidth=2, label=f'POC: ₹{poc:.0f}')
            ax3.axhspan(val, vah, alpha=0.3, color='green', label='Value Area (70%)')
            ax3.set_title("Volume Profile")
            ax3.set_xlabel("Volume")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

        # === EXPORT ===
        result = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "z_score": round(z, 2),
            "RSI_2": round(rsi, 1) if not pd.isna(rsi) else None,
            "VWAP": round(vwap, 0),
            "Trend": "UP" if trend == 1 else "DOWN",
            "POC": round(poc, 0),
            "Signal": reason,
            "Trade_Suggestion": trade.replace("\n", " | ")
        }
        result_df = pd.DataFrame([result])
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="💾 Export Signal to CSV",
            data=csv_buffer.getvalue().encode(),
            file_name="quant_signal.csv",
            mime="text/csv"
        )

        st.success("✅ Analysis complete! All indicators aligned.")

    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        st.info("💡 Tip: Ensure CSV has exact column names (e.g., NIFTY_Close, BANKNIFTY_Volume). Use sample below.")
else:
    st.info("📤 Upload your CSV to analyze. Sample format:")
    sample_data = """Date,Time,NIFTY_Open,NIFTY_High,NIFTY_Low,NIFTY_Close,NIFTY_Volume,BANKNIFTY_Open,BANKNIFTY_High,BANKNIFTY_Low,BANKNIFTY_Close,BANKNIFTY_Volume
2025-11-13,09:15,24800,24820,24790,24810,1500000,51000,51050,50950,51020,1200000"""
    st.code(sample_data, language="csv")
