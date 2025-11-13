# app.py (BULLETPROOF DUAL-FILE VERSION)
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime
import io
import traceback

st.set_page_config(page_title="Bulletproof Dual-File Quant Analyzer", layout="wide")
st.title("Bulletproof Dual-File Indian Quant Analyzer")
st.markdown("**Upload `NIFTY_data.csv` + `BANKNIFTY_data.csv` → Get Signal (Error-Proof)**")

# === CONFIG ===
config = st.sidebar.expander("⚙️ Settings", expanded=True)
with config:
    capital = st.number_input("Capital (₹)", value=500000)
    risk_pct = st.slider("Risk %", 0.5, 5.0, 2.0) / 100
    entry_z = st.slider("Entry z-score", 1.5, 3.0, 2.0)
    n_price = st.number_input("NIFTY FUT Price", value=24850)
    b_price = st.number_input("BANKNIFTY FUT Price", value=51200)

# === FILE UPLOADS ===
nifty_file = st.file_uploader("📁 NIFTY_data.csv (Date,Time,Open,High,Low,Close,Volume)", type="csv")
bank_file = st.file_uploader("📁 BANKNIFTY_data.csv (Date,Time,Open,High,Low,Close,Volume)", type="csv")

if nifty_file and bank_file:
    try:
        # === LOAD ===
        nifty = pd.read_csv(nifty_file)
        bank = pd.read_csv(bank_file)

        # === VALIDATE COLUMNS ===
        req = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        for name, df in [("NIFTY", nifty), ("BANKNIFTY", bank)]:
            missing = [c for c in req if c not in df.columns]
            if missing:
                st.error(f"❌ {name} missing columns: {', '.join(missing)}")
                st.stop()

        # === PARSE DATETIME & VALIDATE LENGTH ===
        nifty['DateTime'] = pd.to_datetime(nifty['Date'] + ' ' + nifty['Time'], errors='coerce')
        bank['DateTime'] = pd.to_datetime(bank['Date'] + ' ' + bank['Time'], errors='coerce')
        
        # Check parsing
        if nifty['DateTime'].isna().any() or bank['DateTime'].isna().any():
            st.error("❌ DateTime parsing failed — check Date/Time format (e.g., 2025-11-13,09:15)")
            st.stop()
        
        if len(nifty) < 10 or len(bank) < 10:
            st.warning("⚠️ Data too short (<10 rows) — Supertrend may be inaccurate.")

        # === MERGE ===
        df = pd.merge(nifty, bank, on='DateTime', suffixes=('_NIFTY', '_BANKNIFTY'), how='inner')
        if len(df) == 0:
            st.error("❌ No matching DateTime rows — ensure times align exactly.")
            st.stop()
        
        df = df.set_index('DateTime').dropna()

        # Rename columns safely
        rename_dict = {
            'Open_NIFTY': 'NIFTY_Open', 'High_NIFTY': 'NIFTY_High', 'Low_NIFTY': 'NIFTY_Low',
            'Close_NIFTY': 'NIFTY_Close', 'Volume_NIFTY': 'NIFTY_Volume',
            'Open_BANKNIFTY': 'BANKNIFTY_Open', 'High_BANKNIFTY': 'BANKNIFTY_High',
            'Low_BANKNIFTY': 'BANKNIFTY_Low', 'Close_BANKNIFTY': 'BANKNIFTY_Close',
            'Volume_BANKNIFTY': 'BANKNIFTY_Volume'
        }
        df.rename(columns=rename_dict, inplace=True)

        # === INDICATORS ===
        # RSI(2) - Safe
        delta = df['NIFTY_Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(2).mean()
        loss = -delta.where(delta < 0, 0).rolling(2).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI_2'] = 100 - (100 / (1 + rs)).fillna(50)  # Fallback to 50 if NaN

        # VWAP - Safe
        df['TPV'] = df['NIFTY_Close'] * df['NIFTY_Volume']
        df['Date'] = df.index.date
        df['CumTPV'] = df.groupby('Date')['TPV'].cumsum()
        df['CumVol'] = df.groupby('Date')['NIFTY_Volume'].cumsum()
        df['VWAP'] = df['CumTPV'] / df['CumVol'].replace(0, np.nan).fillna(method='ffill')

        # Supertrend - Skip if short data
        if len(df) >= 10:
            high, low, close = df['NIFTY_High'], df['NIFTY_Low'], df['NIFTY_Close']
            tr = pd.DataFrame(index=df.index)
            tr['tr0'] = abs(high - low)
            tr['tr1'] = abs(high - close.shift())
            tr['tr2'] = abs(low - close.shift())
            atr = tr.max(axis=1).rolling(7).mean().fillna(method='bfill')
            hl2 = (high + low) / 2
            upper = hl2 + 2.0 * atr
            lower = hl2 - 2.0 * atr
            st = pd.Series(index=df.index, dtype=float)
            trend = pd.Series(index=df.index, dtype=int)
            for i in range(max(7, len(df)//2), len(df)):  # Start later if short
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
            df['Supertrend'] = st.fillna(method='ffill')
            df['Trend'] = trend.fillna(0)
        else:
            df['Trend'] = 0  # Neutral

        # Volume Profile - Safe
        prices = df['NIFTY_Close'].values
        volumes = df['NIFTY_Volume'].values
        if len(prices) > 1:
            hist, edges = np.histogram(prices, bins=min(50, len(prices)//2), weights=volumes)
            poc_idx = np.argmax(hist)
            poc = (edges[poc_idx] + edges[poc_idx + 1]) / 2
            total_vol = volumes.sum()
            cum_vol = 0
            vah, val = poc, poc
            for idx in np.argsort(hist)[::-1]:
                if cum_vol >= total_vol * 0.7:
                    break
                left, right = edges[idx], edges[idx + 1]
                vol = hist[idx]
                cum_vol += vol
                vah = max(vah, right)
                val = min(val, left)
        else:
            poc = vah = val = prices.mean() if len(prices) > 0 else 0

        # Stat Arb - Safe
        y = df['NIFTY_Close'].values
        x = df['BANKNIFTY_Close'].values
        if len(x) > 1:
            slope, _, _, _, _ = stats.linregress(x, y)
            beta = slope
            spread = y - beta * x
            mu = np.mean(spread)
            sigma = np.std(spread)
            if sigma == 0:
                sigma = 1  # Avoid div0
            z = (spread[-1] - mu) / sigma
        else:
            z = 0
            beta = 0.35  # Default

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
            if abs(z) < entry_z: reasons.append("z-score weak")
            if stat_signal == 1 and (pd.isna(rsi) or rsi >= 10): reasons.append("RSI not oversold")
            if stat_signal == -1 and (pd.isna(rsi) or rsi <= 90): reasons.append("RSI not overbought")
            if stat_signal == 1 and price >= vwap: reasons.append("Price ≥ VWAP")
            if stat_signal == -1 and price <= vwap: reasons.append("Price ≤ VWAP")
            if stat_signal == 1 and trend != 1: reasons.append("Supertrend not UP")
            if stat_signal == -1 and trend != -1: reasons.append("Supertrend not DOWN")
            reason = " | ".join(reasons) if reasons else "Data insufficient"

        # Trade Suggestion
        if final_signal != 0:
            risk = capital * risk_pct
            contracts = max(1, int(risk / (abs(z) * 100 * 25 * 0.5)))
            n_qty = contracts * 25
            b_qty = int(contracts * 15 * abs(beta))
            side = "🟢 BUY" if final_signal == 1 else "🔴 SELL"
            opp = "🔴 SELL" if final_signal == 1 else "🟢 BUY"
            trade = f"{side} {n_qty} NIFTY25N27FUT @ ~₹{n_price:,}\n{opp} {b_qty} BANKNIFTY25N27FUT @ ~₹{b_price:,}\nRisk: ₹{int(risk):,} | Target: VAH {'↑' if final_signal==1 else '↓'} ₹{vah:.0f} | Stop: POC ₹{poc:.0f}"
        else:
            trade = "⏸️ HOLD / EXIT — No strong signal"

        # === DISPLAY ===
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
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            ax1.plot(spread, label='Spread', color='purple')
            ax1.axhline(mu, color='green', linestyle='--', label=f'Mean: ₹{mu:.0f}')
            ax1.axhspan(mu - 2*sigma, mu + 2*sigma, alpha=0.2, color='gray')
            ax1.set_title(f"Stat Arb Spread | z = {z:+.2f}")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(df['NIFTY_Close'], label='NIFTY', color='black', alpha=0.7)
            ax2.plot(df['VWAP'], label='VWAP', color='orange', linewidth=2)
            ax2.plot(df['Supertrend'], label='Supertrend', color='green' if trend==1 else 'red', linewidth=2)
            ax2.set_title("NIFTY + VWAP + Supertrend")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        # === EXPORT ===
        result_df = pd.DataFrame([{
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "z_score": round(z, 2),
            "Signal": reason,
            "Trade": trade.replace("\n", " | ")
        }])
        st.download_button(
            label="💾 Export Signal to CSV",
            data=result_df.to_csv(index=False).encode(),
            file_name="quant_signal.csv",
            mime="text/csv"
        )

        st.success(f"✅ Analysis complete! Rows processed: {len(df)}")

    except ValueError as ve:
        st.error(f"❌ Data error: {str(ve)[:100]}... Check CSV format.")
    except pd.errors.MergeError as me:
        st.error("❌ Merge failed: DateTime don't match. Ensure exact times in both CSVs.")
    except Exception as e:
        # Safe error handling - avoid str(e) if Pandas issue
        error_msg = f"Unexpected error: {type(e).__name__} - {str(e)[:100]}..."
        st.error(error_msg)
        st.code(traceback.format_exc()[:200])  # Show trace for debug
else:
    st.info("📤 Upload both CSVs. Sample format:\n**Date**: 2025-11-13 | **Time**: 09:15 | **Open/High/Low/Close/Volume**")
    st.code("""
Date,Time,Open,High,Low,Close,Volume
2025-11-13,09:15,24800,24820,24790,24810,1500000
""", language="csv")
