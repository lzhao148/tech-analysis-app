# streamlit_cycle_scanner_oil_yf.py

import numpy as np
import pandas as pd
import datetime as dt

import streamlit as st
import yfinance as yf
from statsmodels.tsa.filters.hp_filter import hpfilter

# -----------------------------
# Goertzel helpers
# -----------------------------
def goertzel_complex(x, k):
    N = len(x)
    w = 2 * np.pi * k / N
    cosine = np.cos(w)
    sine = np.sin(w)
    coeff = 2 * cosine

    s_prev = 0.0
    s_prev2 = 0.0
    for n in x:
        s = n + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s

    real_part = s_prev - s_prev2 * cosine
    imag_part = s_prev2 * sine
    return real_part + 1j * imag_part

def goertzel_power_amp_phase(x, period):
    N = len(x)
    freq = 1.0 / period
    k = int(round(freq * N))
    if k <= 0 or k >= N:
        return None

    Xk = goertzel_complex(x, k)
    power = (np.abs(Xk) ** 2) / N
    amplitude = 2 * np.abs(Xk) / N
    phase = np.angle(Xk)
    return power, amplitude, phase

# -----------------------------
# Simplified Bartels-like test
# -----------------------------
def bartels_like_score(x, period, phase):
    N = len(x)
    t = np.arange(N)
    sine_wave = np.sin(2 * np.pi * t / period + phase)

    x_std = (x - x.mean()) / (x.std() + 1e-8)
    s_std = (sine_wave - sine_wave.mean()) / (sine_wave.std() + 1e-8)
    rho = np.dot(x_std, s_std) / N
    rho = np.clip(rho, -1, 1)
    score = 1.0 - abs(rho)
    return score

# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(page_title="Cycle Scanner for Oil", layout="wide")

st.title("Cycle Scanner – WTI Oil Example (yfinance)")
st.markdown(
    "Four‑step Cycle Scanner prototype: HP detrending → Goertzel DFT → "
    "Bartels‑style validation → cycle strength ranking.[page:1]"
)

# Sidebar controls
st.sidebar.header("Parameters")

symbol = st.sidebar.text_input("Yahoo symbol", value="CL=F")
start_date = st.sidebar.date_input("Start date", dt.date(2015, 1, 1))
end_date = st.sidebar.date_input("End date", dt.date.today())

hp_lambda = st.sidebar.number_input("HP filter λ", min_value=10.0, max_value=50000.0,
                                    value=1600.0, step=100.0)
min_period = st.sidebar.number_input("Min period", min_value=5, max_value=1000,
                                     value=10, step=1)
max_period = st.sidebar.number_input("Max period", min_value=5, max_value=2000,
                                     value=250, step=5)

genuine_threshold = st.sidebar.slider("Genuine % threshold", 0.0, 100.0, 49.0, 1.0)

run_button = st.sidebar.button("Run Scanner")

# Data fetch using yfinance
@st.cache_data(show_spinner=True)
ef load_data(symbol, start, end):
    # Explicitly disable auto_adjust so 'Adj Close' is present
    df = yf.download(
        symbol,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,   # <-- important
    )
    # Fallback: if 'Adj Close' missing, use 'Close'
    if "Adj Close" in df.columns:
        return df["Adj Close"].dropna()
    else:
        return df["Close"].dropna()
if run_button:
    try:
        prices = load_data(symbol, start_date, end_date)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    st.subheader("Price and HP detrending")

    cycle, trend = hpfilter(prices, lamb=hp_lambda)
    detrended = cycle
    df_plot = pd.DataFrame(
        {"Price": prices, "Trend (HP)": trend, "Detrended (cycle)": detrended}
    )

    st.line_chart(df_plot[["Price", "Trend (HP)"]])
    st.line_chart(df_plot[["Detrended (cycle)"]])

    x = detrended.values.astype(float)
    N = len(x)

    rows = []
    st.subheader("Scanning cycles...")
    progress = st.progress(0)
    periods = list(range(int(min_period), int(max_period) + 1))
    total = len(periods)

    for i, period in enumerate(periods):
        res = goertzel_power_amp_phase(x, period)
        if res is None:
            progress.progress((i + 1) / total)
            continue

        power, amplitude, phase = res
        bartels_score = bartels_like_score(x, period, phase)
        genuine_pct = (1.0 - bartels_score) * 100.0

        if genuine_pct < genuine_threshold:
            progress.progress((i + 1) / total)
            continue

        strength = amplitude / period

        rows.append(
            {
                "period": period,
                "power": power,
                "amplitude": amplitude,
                "phase_rad": phase,
                "bartels_score": bartels_score,
                "genuine_pct": genuine_pct,
                "cycle_strength": strength,
            }
        )
        progress.progress((i + 1) / total)

    if not rows:
        st.warning(
            "No cycles passed the genuineness threshold. "
            "Try lowering the threshold or changing the date range."
        )
        st.stop()

    res_df = pd.DataFrame(rows).sort_values("cycle_strength", ascending=False)

    st.subheader("Top dominant cycles")
    st.dataframe(res_df.head(20))

    st.subheader("Cycle strength vs period")
    st.line_chart(res_df.set_index("period")["cycle_strength"])

    st.download_button(
        "Download cycle table as CSV",
        data=res_df.to_csv(index=False),
        file_name="oil_cycle_scanner_results.csv",
        mime="text/csv",
    )

else:
    st.info("Set parameters on the left and click **Run Scanner** to compute dominant cycles.")
