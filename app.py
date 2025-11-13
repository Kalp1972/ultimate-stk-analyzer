# app.py - FINAL, WITH REAL TRADES
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="NIFTY vs BANKNIFTY Stat Arb", layout="wide")
st.title("NIFTY vs BANKNIFTY Statistical Arbitrage")
st.markdown("**5-min CSVs → Live Signal + Real Backtest + Trade Log**")

nifty_file = st.file_uploader("Upload **NSE_NIFTY, 5_*.csv**", type="csv")
bank_file = st.file_uploader("Upload **NSE_BANKNIFTY, 5_*.csv**", type="csv")

if nifty_file and bank_file:
    try:
        # === LOAD DATA ===
        nifty = pd.read_csv(nifty_file)
        bank = pd.read_csv(bank_file)
        for df_temp in [nifty, bank]:
            df_temp.columns = df_temp.columns.str.strip()

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
        df['RSI_2'] = 100 - (100 / (1 + rs)).fillna(50)

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
        reason = "No edge"
        if z_live < -1.8 and rsi_live < 35 and price < vwap and trend == 1:
            signal = "LONG NIFTY / SHORT BANKNIFTY"
            reason = "z oversold + RSI low + below VWAP + uptrend"
        elif z_live > 1.8 and rsi_live > 65 and price > vwap and trend == -1:
            signal = "SHORT NIFTY / LONG BANKNIFTY"
            reason = "z overbought + RSI high + above VWAP + downtrend"

        st.success(f"**{signal}**")
        st.write(f"**Reason:** {reason}")
        st.write(f"**z:** `{z_live:+.2f}` | **RSI(2):** `{rsi_live:.1f}` | **VWAP:** `₹{vwap:,.0f}`")

        # === BACKTEST ===
        if st.checkbox("Run Full Backtest", value=False):
            st.subheader("Backtest: Stat Arb (5-min)")

            df_bt = df.copy().dropna(subset=['NIFTY_Close', 'BANKNIFTY_Close'])
            window = 50
            n_rows = len(df_bt) - window

            # Pre-allocate
            zs = [0.0] * n_rows
            entry_price_n = [np.nan] * n_rows
            entry_price_b = [np.nan] * n_rows
            exit_price_n = [np.nan] * n_rows
            exit_price_b = [np.nan] * n_rows
            entry_times = [None] * n_rows
            exit_times = [None] * n_rows

            pnl = []
            position = 0
            entry_idx = None
            entry_z = 1.8

            for i in range(window, len(df_bt)):
                idx = i - window
                y = df_bt['NIFTY_Close'].iloc[i-window:i].values
                x = df_bt['BANKNIFTY_Close'].iloc[i-window:i].values

                if len(np.unique(x)) <= 1:
                    beta = 0.44
                    z = 0.0
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

                long_signal = z < -entry_z and rsi < 35 and price < vwap and trend == 1
                short_signal = z > entry_z and rsi > 65 and price > vwap and trend == -1

                if position == 0:
                    if long_signal or short_signal:
                        position = 1 if long_signal else -1
                        entry_idx = idx
                        entry_price_n[idx] = price
                        entry_price_b[idx] = df_bt['BANKNIFTY_Close'].iloc[i]
                        entry_times[idx] = df_bt.index[i]
                else:
                    exit_n = price
                    exit_b = df_bt['BANKNIFTY_Close'].iloc[i]
                    if (position == 1 and z >= -0.5) or (position == -1 and z <= 0.5) or (i - (entry_idx + window) > 20):
                        n_qty = 25
                        b_qty = int(15 * abs(beta))
                        trade_pnl = (
                            (exit_n - entry_price_n[entry_idx]) * n_qty - (exit_b - entry_price_b[entry_idx]) * b_qty
                            if position == 1 else
                            (entry_price_n[entry_idx] - exit_n) * n_qty - (entry_price_b[entry_idx] - exit_b) * b_qty
                        )
                        pnl.append(trade_pnl)
                        exit_price_n[idx] = exit_n
                        exit_price_b[idx] = exit_b
                        exit_times[idx] = df_bt.index[i]
                        position = 0
                        entry_idx = None

                zs[idx] = z

            df_bt = df_bt.iloc[window:window + n_rows].copy()
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

            if num_trades > 0:
                equity = np.cumsum([0] + pnl)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(equity, color='purple', linewidth=2)
                ax.set_title(f"Equity Curve | Final: ₹{equity[-1]:,.0f}")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                st.subheader("Trade Details")
                trade_log = []
                trade_no = 0
                for i in range(len(df_bt)):
                    if not np.isnan(df_bt['entry_n'].iloc[i]) and entry_idx is not None and entry_idx == i:
                        entry_time = df_bt.index[i]
                        entry_n = df_bt['entry_n'].iloc[i]
                        entry_b = df_bt['entry_b'].iloc[i]
                        entry_z = df_bt['z_score'].iloc[i]
                    if not np.isnan(df_bt['exit_n'].iloc[i]):
                        exit_time = df_bt.index[i]
                        exit_n = df_bt['exit_n'].iloc[i]
                        exit_b = df_bt['exit_b'].iloc[i]
                        trade_no += 1
                        trade_log.append({
                            "Trade": trade_no,
                            "Entry": entry_time.strftime("%m-%d %H:%M"),
                            "Exit": exit_time.strftime("%m-%d %H:%M"),
                            "N In": f"₹{entry_n:,.0f}",
                            "B In": f"₹{entry_b:,.0f}",
                            "N Out": f"₹{exit_n:,.0f}",
                            "B Out": f"₹{exit_b:,.0f}",
                            "z": f"{entry_z:+.2f}",
                            "PnL": f"₹{pnl[trade_no-1]:,.0f}"
                        })

                if trade_log:
                    st.dataframe(pd.DataFrame(trade_log), use_container_width=True)
                    st.download_button("Download", pd.DataFrame(trade_log).to_csv(index=False).encode(), "trades.csv")

    except Exception as e:
        st.error(f"Error: {str(e)[:300]}")
else:
    st.info("Upload both CSVs.")
