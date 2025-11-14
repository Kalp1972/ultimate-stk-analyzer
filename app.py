# app.py - FINAL: NIFTY vs BANKNIFTY Pairs Trading (Retail Indian Market)
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="NIFTY vs BANKNIFTY Pairs", layout="wide")
st.title("NIFTY vs BANKNIFTY Pairs Trading")
st.markdown("**5-min CSVs → Live Signal + 12+ Real Trades**")

# === UPLOAD ===
nifty_file = st.file_uploader("**NSE_NIFTY, 5_*.csv**", type="csv")
bank_file = st.file_uploader("**NSE_BANKNIFTY, 5_*.csv**", type="csv")

if nifty_file and bank_file:
    try:
        # === LOAD & CLEAN ===
        nifty = pd.read_csv(nifty_file)
        bank = pd.read_csv(bank_file)
        for d in [nifty, bank]:
            d.columns = d.columns.str.strip()

        nifty['DateTime'] = pd.to_datetime(nifty['Date'] + ' ' + nifty['Time'], errors='coerce')
        bank['DateTime'] = pd.to_datetime(bank['Date'] + ' ' + bank['Time'], errors='coerce')

        nifty = nifty.dropna(subset=['DateTime']).set_index('DateTime').resample('5min').last().ffill()
        bank = bank.dropna(subset=['DateTime']).set_index('DateTime').resample('5min').last().ffill()

        df = pd.concat([nifty.add_prefix('NIFTY_'), bank.add_prefix('BANKNIFTY_')], axis=1).dropna()

        # === INDICATORS ===
        delta = df['NIFTY_Close'].diff()
        gain = delta.clip(lower=0).rolling(2).mean()
        loss = (-delta.clip(upper=0)).rolling(2).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI_2'] = 100 - 100 / (1 + rs)
        df['RSI_2'] = df['RSI_2'].fillna(50)

        df['VWAP'] = (np.cumsum(df['NIFTY_Close'] * df['NIFTY_Volume']) / np.cumsum(df['NIFTY_Volume'])).ffill()
        df['Trend'] = np.where(df['NIFTY_Close'] > df['VWAP'], 1, -1)

        # === Z-SCORE (50-bar rolling) ===
        window = 50
        z_scores = []
        for i in range(len(df)):
            if i < window:
                z_scores.append(0.0)
                continue
            y = df['NIFTY_Close'].iloc[i-window:i]
            x = df['BANKNIFTY_Close'].iloc[i-window:i]
            if len(np.unique(x)) <= 1:
                z_scores.append(0.0)
                continue
            slope, _, _, _, _ = stats.linregress(x, y)
            beta = slope
            spread = y - beta * x
            mu = spread.mean()
            sigma = spread.std() or 1e-6
            z = (spread.iloc[-1] - mu) / sigma
            z_scores.append(z)
        df['z_score'] = z_scores

        # === LIVE SIGNAL ===
        latest = df.iloc[-1]
        z_live = latest['z_score']
        rsi_live = latest['RSI_2']
        price = latest['NIFTY_Close']
        vwap = latest['VWAP']
        trend = latest['Trend']
        time_now = latest.name.time()

        signal = "HOLD"
        reason = "No edge"

        # === RETAIL FILTERS: 9:15 AM to 11:30 AM only ===
        if time_now >= pd.Timestamp("09:15").time() and time_now <= pd.Timestamp("11:30").time():
            if z_live < -1.2 and rsi_live < 45 and price < vwap and trend == 1:
                signal = "LONG NIFTY / SHORT BNF"
                reason = "z oversold + RSI low + below VWAP + uptrend"
            elif z_live > 1.2 and rsi_live > 55 and price > vwap and trend == -1:
                signal = "SHORT NIFTY / LONG BNF"
                reason = "z overbought + RSI high + above VWAP + downtrend"

        st.success(f"**{signal}**")
        st.write(f"**Reason:** {reason}")
        st.write(f"**z:** `{z_live:+.2f}` | **RSI:** `{rsi_live:.1f}` | **VWAP:** `₹{vwap:,.0f}`")

        # === BACKTEST ===
        if st.checkbox("Run Full Backtest", value=False):
            st.subheader("Backtest Results")

            trades = []
            position = 0
            entry_bar = None
            entry_price_n = entry_price_b = None
            entry_time = None
            entry_z_threshold = 1.2  # Slightly tighter for quality

            for i in range(window, len(df)):
                row = df.iloc[i]
                z = row['z_score']
                rsi = row['RSI_2']
                price = row['NIFTY_Close']
                vwap = row['VWAP']
                trend = row['Trend']
                time = row.name.time()

                # === RETAIL FILTER: Only 9:15 AM to 11:30 AM ===
                if time < pd.Timestamp("09:15").time() or time > pd.Timestamp("11:30").time():
                    if position != 0 and (i - entry_bar) > 15:
                        position = 0  # Force exit outside window
                    continue

                long_sig = z < -entry_z_threshold and rsi < 45 and price < vwap and trend == 1
                short_sig = z > entry_z_threshold and rsi > 55 and price > vwap and trend == -1

                if position == 0 and (long_sig or short_sig):
                    position = 1 if long_sig else -1
                    entry_bar = i
                    entry_price_n = price
                    entry_price_b = df['BANKNIFTY_Close'].iloc[i]
                    entry_time = row.name

                if position != 0:
                    exit_price_n = price
                    exit_price_b = df['BANKNIFTY_Close'].iloc[i]
                    revert = (position == 1 and z >= -0.6) or (position == -1 and z <= 0.6)
                    timeout = (i - entry_bar) > 15  # Max 75 mins

                    if revert or timeout:
                        # === FIXED HEDGE RATIO: 1 NIFTY : 1.3 BANKNIFTY ===
                        n_qty = 25
                        b_qty = 33  # 1.3x

                        pnl = (
                            (exit_price_n - entry_price_n) * n_qty - (exit_price_b - entry_price_b) * b_qty
                            if position == 1 else
                            (entry_price_n - exit_price_n) * n_qty - (entry_price_b - exit_price_b) * b_qty
                        )
                        trades.append({
                            "Entry": entry_time.strftime("%m-%d %H:%M"),
                            "Exit": row.name.strftime("%m-%d %H:%M"),
                            "Dir": "LONG N" if position == 1 else "SHORT N",
                            "N In": f"₹{entry_price_n:,.0f}",
                            "B In": f"₹{entry_price_b:,.0f}",
                            "N Out": f"₹{exit_price_n:,.0f}",
                            "B Out": f"₹{exit_price_b:,.0f}",
                            "PnL": f"₹{pnl:,.0f}"
                        })
                        # === FULL RESET ===
                        position = 0
                        entry_bar = None
                        entry_price_n = None
                        entry_price_b = None
                        entry_time = None

            # === RESULTS ===
            total_pnl = sum(float(t["PnL"].replace("₹", "").replace(",", "")) for t in trades)
            num_trades = len(trades)
            win_rate = sum(1 for t in trades if float(t["PnL"].replace("₹", "").replace(",", "")) > 0) / num_trades * 100 if num_trades else 0

            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total PnL", f"₹{total_pnl:,.0f}")
            with col2: st.metric("Trades", num_trades)
            with col3: st.metric("Win Rate", f"{win_rate:.1f}%")
            with col4: st.metric("Avg PnL", f"₹{total_pnl//num_trades:,.0f}" if num_trades else "₹0")

            if num_trades > 0:
                equity = np.cumsum([0] + [float(t["PnL"].replace("₹", "").replace(",", "")) for t in trades])
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(equity, color='green' if equity[-1] > 0 else 'red', linewidth=2)
                ax.set_title(f"Equity Curve | Final: ₹{equity[-1]:,.0f}")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                st.subheader("Trade Log")
                df_trades = pd.DataFrame(trades)
                st.dataframe(df_trades, use_container_width=True)
                st.download_button("Download Log", df_trades.to_csv(index=False).encode(), "trades.csv", "text/csv")
            else:
                st.warning("No trades. Try extending data or relaxing filters.")

    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("Upload both CSVs to start.")
