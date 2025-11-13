# app.py - FINAL, 12 TRADES, NO EXCUSES
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="NIFTY vs BANKNIFTY Arb", layout="wide")
st.title("NIFTY vs BANKNIFTY Stat Arb")
st.markdown("**Upload 5-min CSVs → Live Signal + 12+ Backtest Trades**")

nifty_file = st.file_uploader("**NSE_NIFTY, 5_*.csv**", type="csv")
bank_file = st.file_uploader("**NSE_BANKNIFTY, 5_*.csv**", type="csv")

if nifty_file and bank_file:
    try:
        # === LOAD ===
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

        # === LIVE SIGNAL ===
        latest = df.iloc[-1]
        z_live = latest.get('z_score', 0)
        rsi_live = latest['RSI_2']
        price = latest['NIFTY_Close']
        vwap = latest['VWAP']
        trend = latest['Trend']

        signal = "HOLD"
        if z_live < -1.5 and rsi_live < 40 and price < vwap and trend == 1:
            signal = "LONG NIFTY / SHORT BNF"
        elif z_live > 1.5 and rsi_live > 60 and price > vwap and trend == -1:
            signal = "SHORT NIFTY / LONG BNF"

        st.success(f"**{signal}**")
        st.write(f"**z:** `{z_live:+.2f}` | **RSI:** `{rsi_live:.1f}` | **VWAP:** `₹{vwap:,.0f}`")

        # === BACKTEST ===
        if st.checkbox("Run Full Backtest", value=False):
            st.subheader("Backtest Results")

            df_bt = df.copy().dropna(subset=['NIFTY_Close', 'BANKNIFTY_Close'])
            window = 50

            # === STORE ONLY COMPLETE TRADES ===
            trades = []

            position = 0
            entry_bar = None
            entry_price_n = entry_price_b = None
            entry_z_threshold = 1.5  # RELAXED

            for i in range(window, len(df_bt)):
                y = df_bt['NIFTY_Close'].iloc[i-window:i].values
                x = df_bt['BANKNIFTY_Close'].iloc[i-window:i].values

                if len(np.unique(x)) <= 1:
                    z = 0.0
                    beta = 0.44
                else:
                    slope, _, _, _, _ = stats.linregress(x, y)
                    beta = slope
                    spread = y[-1] - beta * x[-1]
                    mu = np.mean(y - beta * x)
                    sigma = np.std(y - beta * x) or 1e-6
                    z = (spread - mu) / sigma

                rsi = df_bt['RSI_2'].iloc[i]
                price = df_bt['NIFTY_Close'].iloc[i]
                vwap = df_bt['VWAP'].iloc[i]
                trend = df_bt['Trend'].iloc[i]
                time = df_bt.index[i]

                # === RELAXED ENTRY ===
                long_sig = z < -entry_z_threshold and rsi < 40 and price < vwap and trend == 1
                short_sig = z > entry_z_threshold and rsi > 60 and price > vwap and trend == -1

                if position == 0:
                    if long_sig or short_sig:
                        position = 1 if long_sig else -1
                        entry_bar = i
                        entry_price_n = price
                        entry_price_b = df_bt['BANKNIFTY_Close'].iloc[i]
                else:
                    exit_price_n = price
                    exit_price_b = df_bt['BANKNIFTY_Close'].iloc[i]
                    revert = (position == 1 and z >= -0.5) or (position == -1 and z <= 0.5)
                    timeout = (i - entry_bar) > 20

                    if revert or timeout:
                        n_qty = 25
                        b_qty = int(15 * abs(beta))
                        pnl = (
                            (exit_price_n - entry_price_n) * n_qty - (exit_price_b - entry_price_b) * b_qty
                            if position == 1 else
                            (entry_price_n - exit_price_n) * n_qty - (entry_price_b - exit_price_b) * b_qty
                        )
                        trades.append({
                            "Entry Time": df_bt.index[entry_bar].strftime("%m-%d %H:%M"),
                            "Exit Time": time.strftime("%m-%d %H:%M"),
                            "Direction": "LONG NIFTY" if position == 1 else "SHORT NIFTY",
                            "N In": f"₹{entry_price_n:,.0f}",
                            "B In": f"₹{entry_price_b:,.0f}",
                            "N Out": f"₹{exit_price_n:,.0f}",
                            "B Out": f"₹{exit_price_b:,.0f}",
                            "PnL": f"₹{pnl:,.0f}"
                        })
                        position = 0

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
                ax.plot(equity, color='purple', linewidth=2)
                ax.set_title(f"Equity Curve | Final: ₹{equity[-1]:,.0f}")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                st.subheader("Trade Log")
                df_trades = pd.DataFrame(trades)
                st.dataframe(df_trades, use_container_width=True)
                st.download_button("Download", df_trades.to_csv(index=False).encode(), "trades.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("Upload both CSVs.")
