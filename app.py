"""
AI-Powered Risk Management Dashboard — Beverage Production Line
NUST MEng Research · Mark Mbewe (N02534127N) · Supervisor: Eng T. Muhla

Advanced Real-Time Version: Features a 1s auto-refreshing simulation engine, 
historical statistical learning, Gaussian random walk drifts, and full-scale 
time-series tracking for all 6 machines.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Risk Dashboard — Beverage Line | NUST",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Trigger automatic rerun every 1000 milliseconds (1 second)
# This drives the live clock and the rolling simulation pipeline.
refresh_count = st_autorefresh(interval=1000, key="prod_line_refresh")

# ─────────────────────────────────────────────
# GLOBAL DARK THEME CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main, .block-container {
    background-color: #060c14 !important;
    color: #c8dff0 !important;
  }
  [data-testid="stSidebar"] {
    background: #0b1623 !important;
    border-right: 1px solid #1a3050;
  }
  [data-testid="stSidebar"] * { color: #c8dff0 !important; }
  h1,h2,h3,h4,h5,h6,p,label,div { color: #c8dff0; }
  .metric-card {
    background: #0f1b2d;
    border: 1px solid #1e3556;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 10px;
  }
  .machine-header {
    background: #12253f;
    padding: 8px 12px;
    border-radius: 4px;
    margin-top: 15px;
    font-weight: bold;
    border-left: 4px solid #00a8ff;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING & STATISTICAL KNOWLEDGE BASE
# ─────────────────────────────────────────────
DATA_PATH = "MARK_MBEWE_DATA_SET.csv"

@st.cache_data
def load_and_analyze_dataset(path):
    df = pd.read_csv(path)
    # Mapping exact columns from the uploaded dataset structure
    machine_metrics = {
        "KHS Innofill": ["khs_innofill_speed_bph", "khs_innofill_product_temp_C", "khs_innofill_co2_pressure_bar"],
        "Krones Checkmat": ["krones_checkmat_throughput_bph", "krones_checkmat_reject_rate_pct", "krones_checkmat_camera_temp_C"],
        "Sidel SBO Blow": ["sidel_sbo_blow_cycle_time_s", "sidel_sbo_oven_temp_C", "sidel_sbo_stretch_pressure_bar", "sidel_sbo_output_rate_bph"],
        "Tetra Pak Pasteurizer": ["tetrapak_past_inlet_temp_C", "tetrapak_past_outlet_temp_C", "tetrapak_past_hold_time_s", "tetrapak_past_flow_rate_Lhr"],
        "Krones Variopac": ["krones_variopac_film_tension_N", "krones_variopac_tunnel_temp_C", "krones_variopac_speed_packs_min", "krones_variopac_film_remaining_pct"],
        "Heuft Spectrum": ["heuft_spectrum_belt_speed_ms", "heuft_spectrum_motor_current_A", "heuft_spectrum_vibration_mms", "heuft_spectrum_jam_count"]
    }
    
    # Calculate means and standard deviations from history to drive the engine
    stats = {}
    for machine, columns in machine_metrics.items():
        stats[machine] = {}
        for col in columns:
            if col in df.columns:
                stats[machine][col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()) if df[col].std() > 0 else 1.0,
                    "min": float(df[col].min()),
                    "max": float(df[col].max())
                }
    return df, stats, machine_metrics

try:
    df_historical, historical_stats, MACHINE_COLUMNS = load_and_analyze_dataset(DATA_PATH)
except Exception as e:
    st.error(f"Failed to process CSV file at {DATA_PATH}. Error: {e}")
    st.stop()

# ─────────────────────────────────────────────
# SIMULATION ENGINE (STATE PRESERVATION)
# ─────────────────────────────────────────────
# Initialize rolling session memory for up to 60 historical ticks
MAX_HISTORY = 60

if "sim_history" not in st.session_state:
    # Bootstrap with the last row of the historical dataset to ground the start state
    last_row = df_historical.iloc[-1]
    initial_state = {"timestamp": datetime.now().strftime("%H:%M:%S")}
    
    for machine, metrics in MACHINE_COLUMNS.items():
        for metric in metrics:
            initial_state[metric] = float(last_row[metric]) if metric in df_historical.columns else historical_stats[machine][metric]["mean"]
            
    st.session_state.sim_history = [initial_state]

# Simulation Parameters controlled via Sidebar
st.sidebar.header("⚙️ Simulation Settings")
drift_factor = st.sidebar.slider("Gaussian Noise Drift Scale (σ)", 0.05, 0.50, 0.15, step=0.05)
anomaly_bias = st.sidebar.slider("System Stress Factor (Bias)", -0.1, 0.1, 0.0, step=0.02)

# Generate Next Step via Gaussian Random Walk Process
last_tick_state = st.session_state.sim_history[-1]
new_tick_state = {"timestamp": datetime.now().strftime("%H:%M:%S")}

for machine, metrics in MACHINE_COLUMNS.items():
    for metric in metrics:
        meta = historical_stats[machine][metric]
        current_val = last_tick_state[metric]
        
        # Gaussian Noise calculation scaled by historical variance and user input
        noise = np.random.normal(loc=anomaly_bias * meta["std"], scale=drift_factor * meta["std"])
        updated_val = current_val + noise
        
        # Hard physical boundary constraints to prevent runaway numbers
        updated_val = max(meta["min"] * 0.8, min(meta["max"] * 1.2, updated_val))
        new_tick_state[metric] = round(updated_val, 3)

# Append new state and trim window size
st.session_state.sim_history.append(new_tick_state)
if len(st.session_state.sim_history) > MAX_HISTORY:
    st.session_state.sim_history.pop(0)

# Convert active rolling window to data frame for charting
df_sim = pd.DataFrame(st.session_state.sim_history)

# ─────────────────────────────────────────────
# HEADER BANNER & TICKING LIVE CLOCK
# ─────────────────────────────────────────────
col_title, col_clock = st.columns([3, 1])
with col_title:
    st.title("⚙️ AI Beverage Production Risk Core")
    st.caption("NUST MEng Research • Candidate: Mark Mbewe (N02534127N) • Supervisor: Eng T. Muhla")

with col_clock:
    # Dynamic live ticking display updating every second via autorefresh
    current_time = datetime.now()
    st.markdown(f"""
    <div style="text-align: right; background: #0f1b2d; padding: 10px; border-radius: 6px; border: 1px solid #1a3050;">
        <span style="color: #8ab4f8; font-size: 0.85rem; font-weight: bold; letter-spacing: 1px;">LIVE SYSTEM CLOCK</span><br>
        <span style="font-family: monospace; font-size: 1.5rem; color: #00a8ff; font-weight: bold;">
            {current_time.strftime('%Y-%m-%d %H:%M:%S')}
        </span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# MACHINE TIME-SERIES AND STATE MONITORS
# ─────────────────────────────────────────────
st.subheader("🏭 Live Machine Telemetry & Real-Time Drifts")

# Iteratively render UI layers across columns for all 6 target manufacturing blocks
m_list = list(MACHINE_COLUMNS.keys())

def render_machine_block(machine_name):
    st.markdown(f"<div class='machine-header'>{machine_name}</div>", unsafe_allow_html=True)
    metrics = MACHINE_COLUMNS[machine_name]
    
    # Showcase current instantaneous metrics
    latest_metrics = df_sim.iloc[-1]
    metric_cols = st.columns(len(metrics))
    
    for idx, col in enumerate(metrics):
        with metric_cols[idx]:
            # Clean variable label display
            display_label = col.replace(machine_name.lower().replace(" ", "_") + "_", "").replace("_", " ").title()
            val = latest_metrics[col]
            
            # Simple color threshold validation based on historical bounds
            hist_meta = historical_stats[machine_name][col]
            is_anomaly = val > hist_meta["max"] or val < hist_meta["min"]
            val_color = "#ff4b4b" if is_anomaly else "#00e676"
            
            st.markdown(f"""
            <div class="metric-card">
                <span style="font-size:0.8rem; color:#8ab4f8;">{display_label}</span><br>
                <span style="font-size:1.3rem; font-family:monospace; color:{val_color}; font-weight:bold;">{val}</span>
            </div>
            """, unsafe_allow_html=True)
            
    # Chart entire rolling history sequence
    chart_data = df_sim[["timestamp"] + metrics].set_index("timestamp")
    st.line_chart(chart_data, height=180, use_container_width=True)

# Grid Layout for the 6 production units
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    render_machine_block(m_list[0]) # KHS Innofill
with row1_col2:
    render_machine_block(m_list[1]) # Krones Checkmat

st.markdown("---")

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    render_machine_block(m_list[2]) # Sidel SBO Blow
with row2_col2:
    render_machine_block(m_list[3]) # Tetra Pak Pasteurizer

st.markdown("---")

row3_col1, row3_col2 = st.columns(2)
with row3_col1:
    render_machine_block(m_list[4]) # Krones Variopac
with row3_col2:
    render_machine_block(m_list[5]) # Heuft Spectrum

# ─────────────────────────────────────────────
# SIMULATED INTERACTIVE DATASET EXPLORER
# ─────────────────────────────────────────────
st.markdown("---")
with st.expander("📊 Live Simulation Buffer Explorer"):
    st.markdown("This dataframe showcases the active rolling memory buffer currently managing the Gaussian Random Walk changes.")
    st.dataframe(df_sim, use_container_width=True)