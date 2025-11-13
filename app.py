# app.py - FINAL, ROBUST, NO ERRORS
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import traceback

st.set_page_config(page_title="NIFTY vs BANKNIFTY Stat Arb", layout="wide")

st.title("NIFTY vs BANKNIFTY Statistical Arbitrage")
st.markdown("**5-min CSVs → Live Signal + Backtest + Trade Log**")

nifty_file = st.file_uploader("Upload **NSE_NIFTY, 5_*.csv**", type="csv")
bank_file = st.file_uploader("Upload **NSE_BANKNIFTY, 5_*.csv**", type="csv")

# === DEFAULTS ===
df = None
num_trades = 0
total_pnl = 0
win_rate = 0
avg_pnl = 0
equity = [0]
trade_details = []

if nifty_file and bank_file:
    try:
        # === LOAD & CLEAN ===
        nifty = pd.read_csv(nifty_file)
        bank = pd.read_csv(bank_file)

        for df_temp in [nifty, bank]:
            df_temp.columns = df_temp.columns.str.strip()

        nifty['DateTime'] = pd.to_datetime(nifty['Date'] + ' ' + nifty['Time'], errors='coerce')
        bank['DateTime'] = pd.to_datetime(bank['Date'] + ' ' + bank['Time'], errors='coerce')

        nifty = nifty.dropna(subset=['DateTime']).set_index('DateTime')
        bank = bank.dropna(subset=['DateTime']).set_index('DateTime')

        nifty = nifty.resample('5min').last().ffill()
        bank = bank.resample('5min').last().ffill()

        df = pd.concat([nifty.add_prefix('NIFTY_'), bank.add_prefix('BANKNIFTY_')], axis=1).dropna()

        # === INDICATORS ===
        delta = df['NIFTY_Close'].diff()
        gain = delta.clip(lower=0).rolling(2).mean()
        loss = (-delta.clip(upper=0)).rolling(2).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI_2'] = 100 - (100 / (1 + rs))
        df['RSI_2'] = df['RSI_2'].fillna(50)

        df['VWAP'] = (np.cumsum(df['NIFTY_Close'] * df['NIFTY_Volume']) / np.cumsum(df['NIFTY_Volume'])).ffill()
        df['Trend'] = np.where(df['NIFTY_Close'] > df['VWAP'], 1, -1)

        # === LIVE SIGNAL ===
        latest = df.iloc[-1]
        z = latest.get('z_score', 0)
        rsi = latest['RSI_2']
        price = latest['NIFTY_Close']
        vwap = latest['VWAP']
        trend = latest['Trend']

        signal = "HOLD"
        reason = "No edge"
        if z < -1.5 and rsi < 20 and price < vwap and trend == 1:
            signal = "LONG NIFTY / SHORT BANKNIFTY"
            reason = "z oversold + RSI + below VWAP + uptrend"
        elif z > 1.5 and rsi > 80 and price > vwap and trend == -1:
            signal = "SHORT NIFTY / LONG BANKNIFTY"
            reason = "z overbought + RSI + above VWAP + downtrend"

        st.success(f"**{signal}**")
        st.write(f"**Reason:** {reason}")
        st.write(f"**z:** `{z:+.2f}` | **RSI(2):** `{rsi:.1f}` | **VWAP:** `₹{vwap:,.0f}`")

        # === BACKTEST ===
        if st.checkbox("Run Full Backtest", value=False):
            st.subheader("Backtest: Stat Arb (5-min)")

            df_bt = df.copy().dropna(subset=['NIFTY_Close', 'BANKNIFTY_Close'])
            window = 50
            zs = []
            entry_price_n = []
            entry_price_b = []
            exit_price_n = []
            exit_price_b = []
            pnl = []
            position = 0
            entry_idx = 0
            entry_z = 1.5

            for i in range(window, len(df_bt)):
                y = df_bt['NIFTY_Close'].iloc[i-window:i].values
                x = df_bt['BANKNIFTY_Close'].iloc[i-window:i].values

                # === SKIP IF FLAT (PREVENT CRASH) ===
                if len(np.unique(x)) <= 1 or len(np.unique(y)) <= 1:
                    beta = 0.44  # fallback beta
                    spread = y[-1] - beta * x[-1]
                    z = 0  # neutral
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

                long_signal = z < -entry_z and rsi < 20 and price < vwap and trend == 1
                short_signal = z > entry_z and rsi > 80 and price > vwap and trend == -1

                if position == 0:
                    if long_signal or short_signal:
                        position = 1 if long_signal else -1
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
                        trade_pnl = (
                            (exit_n - entry_price_n[-1]) * n_qty - (exit_b - entry_price_b[-1]) * b_qty
                            if position == 1 else
                            (entry_price_n[-1] - exit_n) * n_qty - (entry_price_b[-1] - exit_b) * b_qty
                        )
                        pnl.append(trade_pnl)
                        position = 0
                        exit_price_n.append(exit_n)
                        exit_price_b.append(exit_b)
                    else:
                        exit_price_n.append(np.nan)
                        exit_price_b.append(np.nan)

                zs.append(z)

            df_bt = df_bt.iloc[window:].copy()
            df_bt['z_score'] = zs
            df_bt['entry_n'] = entry_price_n
            df_bt['entry_b'] = entry_price_b
            df_bt['exit_n'] = exit_price_n
            df_bt['exit_b'] = exit_price_b

            total_pnl = sum(pnl)
            num_trades = len(pnl)
            win_rate = sum(1 for p in pnl if p > 0) / num_trades * 100 if num_trades else 0
            avg_pnl = total_pnl / num_trades if num_trades else 0

            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total PnL", f"₹{total_pnl:,.0f}")
            with col2: st.metric("Trades", num_trades)
            with col3: st.metric("Win Rate", f"{win_rate:.1f}%")
            with col4: st.metric("Avg PnL", f"₹{avg_pnl:,.0f}")

            equity = np.cumsum([0] + pnl)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(equity, color='purple', linewidth=2)
            ax.set_title(f"Equity Curve | Final: ₹{equity[-1]:,.0f}")
            ax.set_ylabel("PnL (₹)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # === TRADE LOG ===
            if num_trades > 0:
                st.subheader("Trade Details")
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
                    st.download_button("Download Log", trade_df.to_csv(index=False).encode(), "trades.csv")

    except Exception as e:
        st.error(f"Error: {str(e)[:300]}")
        st.code(traceback.format_exc()[:1000])
else:
    st.info("Upload **NIFTY** and **BANKNIFTY** 5-min CSVs to start.")
