# app.py - ULTIMATE NIFTY vs BANKNIFTY STAT ARB ANALYZER
# With Live Signal + Full Backtest + Trade-by-Trade Log

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import matplotlib.pyplot as plt
import traceback

st.set_page_config(page_title="NIFTY vs BANKNIFTY Stat Arb", layout="wide")

# === TITLE ===
st.title("NIFTY vs BANKNIFTY Statistical Arbitrage")
st.markdown("Upload **5-min CSVs** → Get **Live Signal + Full Backtest + Trade Log**")

# === FILE UPLOAD ===
nifty_file = st.file_uploader("Upload **NSE_NIFTY, 5_*.csv**", type="csv")
bank_file = st.file_uploader("Upload **NSE_BANKNIFTY, 5_*.csv**", type="csv")

# === INITIALIZE VARIABLES SAFELY ===
df = None
num_trades = 0
total_pnl = 0
win_rate = 0
avg_pnl = 0
equity = [0]
trade_details = []

if nifty_file and bank_file:
    try:
        # === LOAD & PARSE DATA ===
        nifty = pd.read_csv(nifty_file)
        bank = pd.read_csv(bank_file)

        # Clean column names
        nifty.columns = nifty.columns.str.strip()
        bank.columns = bank.columns.str.strip()

        # Parse datetime
        nifty['DateTime'] = pd.to_datetime(nifty['Date'] + ' ' + nifty['Time'], format='%Y-%m-%dT%H:%M:%S%z', errors='coerce')
        bank['DateTime'] = pd.to_datetime(bank['Date'] + ' ' + bank['Time'], format='%Y-%m-%dT%H:%M:%S%z', errors='coerce')

        # Drop invalid
        nifty = nifty.dropna(subset=['DateTime'])
        bank = bank.dropna(subset=['DateTime'])

        # Set index
        nifty.set_index('DateTime', inplace=True)
        bank.set_index('DateTime', inplace=True)

        # Resample to 5-min aligned
        nifty = nifty.resample('5min').last().ffill()
        bank = bank.resample('5min').last().ffill()

        # Merge
        df = pd.concat([nifty.add_prefix('NIFTY_'), bank.add_prefix('BANKNIFTY_')], axis=1)
        df = df.dropna()

        # === INDICATORS ===
        # RSI(2)
        delta = df['NIFTY_Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(2).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI_2'] = 100 - (100 / (1 + rs))
        df['RSI_2'] = df['RSI_2'].fillna(50)

        # VWAP
        q = df['NIFTY_Volume']
        p = df['NIFTY_Close']
        df['VWAP'] = (np.cumsum(p * q) / np.cumsum(q)).fillna(method='ffill')

        # Trend (simplified)
        df['Trend'] = np.where(df['NIFTY_Close'] > df['VWAP'], 1, -1)

        # === LIVE SIGNAL (LAST ROW) ===
        latest = df.iloc[-1]
        z = latest.get('z_score', 0)
        rsi = latest['RSI_2']
        price = latest['NIFTY_Close']
        vwap = latest['VWAP']
        trend = latest['Trend']

        signal = "HOLD / EXIT — No strong signal"
        reason = "z-score weak"
        if z < -1.5 and rsi < 20 and price < vwap and trend == 1:
            signal = "LONG NIFTY / SHORT BANKNIFTY"
            reason = "z-score oversold + RSI + below VWAP + uptrend"
        elif z > 1.5 and rsi > 80 and price > vwap and trend == -1:
            signal = "SHORT NIFTY / LONG BANKNIFTY"
            reason = "z-score overbought + RSI + above VWAP + downtrend"

        st.success(f"**{signal}**")
        st.write(f"**Reason:** {reason}")
        st.write(f"**z-score:** `{z:+.2f}` | **RSI(2):** `{rsi:.1f}` | **VWAP:** `₹{vwap:,.0f}` | **Trend:** `{'UP' if trend == 1 else 'DOWN'}`")

        # === BACKTEST ===
        if st.checkbox("Run Full Backtest (Oct 15 – Nov 13, 2025)", value=False):
            st.subheader("Backtest: NIFTY vs BANKNIFTY Stat Arb (5-min)")

            df_bt = df.copy()
            df_bt = df_bt.dropna(subset=['NIFTY_Close', 'BANKNIFTY_Close'])

            window = 50
            spreads = []
            zs = []
            signals = []
            entry_price_n = []
            entry_price_b = []
            exit_price_n = []
            exit_price_b = []
            pnl = []
            position = 0
            entry_idx = 0
            entry_z = 1.5

            for i in range(window, len(df_bt)):
                window_data = df_bt.iloc[i-window:i]
                y = window_data['NIFTY_Close'].values
                x = window_data['BANKNIFTY_Close'].values
                slope, _, _, _, _ = stats.linregress(x, y)
                beta = slope
                spread = y[-1] - beta * x[-1]
                mu = np.mean(y - beta * x)
                sigma = np.std(y - beta * x)
                if sigma == 0: sigma = 1e-6
                z = (spread - mu) / sigma

                rsi = df_bt['RSI_2'].iloc[i]
                price = df_bt['NIFTY_Close'].iloc[i]
                vwap = df_bt['VWAP'].iloc[i]
                trend = df_bt['Trend'].iloc[i]

                long_signal = z < -entry_z and rsi < 20 and price < vwap and trend == 1
                short_signal = z > entry_z and rsi > 80 and price > vwap and trend == -1

                if position == 0:
                    if long_signal:
                        position = 1
                        entry_idx = i
                        entry_price_n.append(price)
                        entry_price_b.append(df_bt['BANKNIFTY_Close'].iloc[i])
                    elif short_signal:
                        position = -1
                        entry_idx = i
                        entry_price_n.append(price)
                        entry_price_b.append(df_bt['BANKNIFTY_Close'].iloc[i])
                    else:
                        entry_price_n.append(np.nan)
                        entry_price_b.append(np.nan)
                else:
                    exit_n = price
                    exit_b = df_bt['BANKNIFTY_Close'].iloc[i]
                    if (position == 1 and z >= -0.5) or (position == -1 and z <= 0.5) or (i - entry_idx > 20):
                        n_qty = 25
                        b_qty = int(15 * abs(beta))
                        if position == 1:
                            trade_pnl = (exit_n - entry_price_n[-1]) * n_qty - (exit_b - entry_price_b[-1]) * b_qty
                        else:
                            trade_pnl = (entry_price_n[-1] - exit_n) * n_qty - (entry_price_b[-1] - exit_b) * b_qty
                        pnl.append(trade_pnl)
                        position = 0
                        exit_price_n.append(exit_n)
                        exit_price_b.append(exit_b)
                    else:
                        exit_price_n.append(np.nan)
                        exit_price_b.append(np.nan)

                spreads.append(spread)
                zs.append(z)
                signals.append(long_signal or short_signal)

            df_bt = df_bt.iloc[window:].copy()
            df_bt['z_score'] = zs
            df_bt['spread'] = spreads
            df_bt['signal'] = signals
            df_bt['entry_n'] = entry_price_n
            df_bt['entry_b'] = entry_price_b
            df_bt['exit_n'] = exit_price_n
            df_bt['exit_b'] = exit_price_b

            # === METRICS ===
            total_pnl = sum(p for p in pnl if not np.isnan(p))
            num_trades = len(pnl)
            win_rate = len([p for p in pnl if p > 0]) / num_trades * 100 if num_trades else 0
            avg_pnl = total_pnl / num_trades if num_trades else 0

            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total PnL", f"₹{total_pnl:,.0f}")
            with col2: st.metric("Trades", num_trades)
            with col3: st.metric("Win Rate", f"{win_rate:.1f}%")
            with col4: st.metric("Avg PnL", f"₹{avg_pnl:,.0f}")

            # === EQUITY CURVE ===
            equity = np.cumsum([0] + pnl)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(equity, color='purple', linewidth=2)
            ax.set_title(f"Equity Curve | Final: ₹{equity[-1]:,.0f}")
            ax.set_ylabel("Cumulative PnL (₹)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # === DETAILED TRADE LOG ===
            if num_trades > 0:
                st.subheader("Trade-by-Trade Breakdown")
                trade_details = []
                trade_idx = 0
                entry_time = entry_n = entry_b = entry_z_val = beta_val = direction = None

                for i in range(len(df_bt)):
                    row = df_bt.iloc[i]
                    if position == 0 and not np.isnan(row['entry_n']):
                        entry_time = row.name
                        entry_n = row['entry_n']
                        entry_b = row['entry_b']
                        entry_z_val = row['z_score']
                        direction = "LONG NIFTY / SHORT BNF" if row['z_score'] < -entry_z else "SHORT NIFTY / LONG BNF"
                        beta_val = row['NIFTY_Close'] / row['BANKNIFTY_Close']
                    elif position != 0 and not np.isnan(row['exit_n']):
                        exit_time = row.name
                        exit_n = row['exit_n']
                        exit_b = row['exit_b']
                        n_qty = 25
                        b_qty = int(15 * abs(beta_val))
                        trade_pnl = (
                            (exit_n - entry_n) * n_qty - (exit_b - entry_b) * b_qty
                            if "LONG" in direction else
                            (entry_n - exit_n) * n_qty - (entry_b - exit_b) * b_qty
                        )
                        trade_details.append({
                            "Trade": trade_idx + 1,
                            "Entry": entry_time.strftime("%m-%d %H:%M"),
                            "Exit": exit_time.strftime("%m-%d %H:%M"),
                            "Dir": direction.split()[0],
                            "N In": f"₹{entry_n:,.0f}",
                            "B In": f"₹{entry_b:,.0f}",
                            "N Out": f"₹{exit_n:,.0f}",
                            "B Out": f"₹{exit_b:,.0f}",
                            "z": f"{entry_z_val:+.2f}",
                            "PnL": f"₹{trade_pnl:,.0f}"
                        })
                        trade_idx += 1

                if trade_details:
                    trade_df = pd.DataFrame(trade_details)
                    st.dataframe(trade_df, use_container_width=True)
                    st.download_button(
                        "Download Trade Log",
                        trade_df.to_csv(index=False).encode(),
                        "trade_details.csv",
                        "text/csv"
                    )

    except Exception as e:
        st.error(f"Error: {str(e)[:300]}")
        st.code(traceback.format_exc()[:1000])

else:
    st.info("Please upload both **NIFTY** and **BANKNIFTY** 5-min CSVs to begin.")
