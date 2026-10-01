import json
import os
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
import streamlit as st
from dotenv import load_dotenv

# ---------------- Config ----------------
load_dotenv("ayush.env")
BROKER = os.getenv("MQTT_BROKER")
PORT = int(os.getenv("MQTT_PORT", 8883))
USERNAME = os.getenv("MQTT_USERNAME")
PASSWORD = os.getenv("MQTT_PASSWORD")
TOPIC = os.getenv("MQTT_TOPIC")

st.set_page_config(page_title="Battery Pack Monitor", page_icon="🔋", layout="wide")
st.title("🔋 Battery Pack Monitor")


# ---------------- MQTT (connected once, reused across reruns) ----------------
@st.cache_resource
def get_mqtt_client():
    if not all([BROKER, USERNAME, PASSWORD, TOPIC]):
        raise ValueError("Missing MQTT_* values in ayush.env")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set()
    client.connect(BROKER, PORT, 60)
    client.loop_start()
    return client


# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    pack_id = st.text_input("Pack ID", "PACK_01")
    num_cells = st.slider("Number of cells", 2, 12, 4)
    base_voltage = st.number_input("Base cell voltage (V)", 3.0, 4.2, 3.90, 0.01)
    interval_sec = st.slider("Update interval (sec)", 1, 10, 2)
    max_points = st.slider("Points kept in charts", 50, 1000, 300, 50)

    st.subheader("Imbalance simulation")
    imbalance_mode = st.toggle("Imbalance mode", value=True)
    weak_cell = st.selectbox("Weak cell", list(range(1, num_cells + 1)), index=num_cells - 1)
    drop_rate = st.number_input("Drop per step (V)", 0.0, 0.01, 0.0005, 0.0001, format="%.4f")

    st.subheader("Alert thresholds (V_delta)")
    warn_th = st.number_input("Warning (V)", 0.005, 0.2, 0.020, 0.005, format="%.3f")
    crit_th = st.number_input("Critical (V)", 0.01, 0.5, 0.050, 0.005, format="%.3f")

    st.subheader("Control")
    publish_mqtt = st.toggle("Publish to HiveMQ", value=True)
    running = st.toggle("Running", value=True)
    if st.button("🗑️ Reset data", use_container_width=True):
        st.session_state.history = []
        st.session_state.weak_drop = 0.0
        st.rerun()

# ---------------- Session state ----------------
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.weak_drop = 0.0

mqtt_client = None
if publish_mqtt:
    try:
        mqtt_client = get_mqtt_client()
        st.sidebar.success("Connected to HiveMQ")
    except Exception as e:
        st.sidebar.error(f"MQTT error: {e}")

cell_cols = [f"Cell {i + 1}" for i in range(num_cells)]


# ---------------- Data generation ----------------
def generate_reading():
    ss = st.session_state
    cells = [round(random.normalvariate(base_voltage, 0.005), 3) for _ in range(num_cells)]

    if imbalance_mode:
        ss.weak_drop += drop_rate
        idx = weak_cell - 1
        cells[idx] = round(cells[idx] - ss.weak_drop, 3)

    now = datetime.now(timezone.utc)
    payload = {
        "pack_id": pack_id,
        "timestamp": now.isoformat(),
        "cell_v": cells,
        "pack_current": 1.2,
        "temp_c": 30.0,
    }
    if mqtt_client is not None:
        mqtt_client.publish(TOPIC, json.dumps(payload))

    row = {"time": now, "pack_v": round(sum(cells), 3), "v_delta": round(max(cells) - min(cells), 3)}
    row.update(dict(zip(cell_cols, cells)))
    ss.history.append(row)
    ss.history = ss.history[-max_points:]


def health_status(v_delta):
    if v_delta >= crit_th:
        return "critical"
    if v_delta >= warn_th:
        return "warning"
    return "ok"


# ---------------- Live dashboard (auto-refreshes without blocking) ----------------
@st.fragment(run_every=interval_sec if running else None)
def live_dashboard():
    if running:
        generate_reading()

    if not st.session_state.history:
        st.info("Waiting for data... turn on 'Running' in the sidebar.")
        return

    df = pd.DataFrame(st.session_state.history)
    last = df.iloc[-1]
    latest_cells = last[cell_cols].astype(float)
    weakest = latest_cells.idxmin()
    status = health_status(last["v_delta"])

    # --- Metrics ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pack voltage", f"{last['pack_v']:.3f} V")
    c2.metric("V_delta", f"{last['v_delta'] * 1000:.1f} mV",
              delta=f"{(last['v_delta'] - df['v_delta'].iloc[-2]) * 1000:.1f} mV" if len(df) > 1 else None,
              delta_color="inverse")
    c3.metric("Max cell", f"{latest_cells.max():.3f} V", latest_cells.idxmax())
    c4.metric("Min cell", f"{latest_cells.min():.3f} V", weakest, delta_color="off")
    c5.metric("Samples", len(df))

    # --- Alert banner ---
    if status == "critical":
        st.error(f"🚨 CRITICAL imbalance: {weakest} is {last['v_delta'] * 1000:.0f} mV below the strongest cell.")
    elif status == "warning":
        st.warning(f"⚠️ Warning: imbalance rising, weakest cell is {weakest}.")
    else:
        st.success("✅ Pack is balanced.")

    tab1, tab2, tab3 = st.tabs(["📈 Live charts", "🔍 Analysis", "🗂️ Raw data"])

    # --- Tab 1: charts ---
    with tab1:
        st.subheader("Cell voltages over time")
        st.line_chart(df.set_index("time")[cell_cols], height=320)

        left, right = st.columns(2)
        with left:
            st.subheader("V_delta over time (V)")
            st.line_chart(df.set_index("time")[["v_delta"]], height=250)
        with right:
            st.subheader("Cell deviation from pack average (mV)")
            dev = (latest_cells - latest_cells.mean()) * 1000
            st.bar_chart(dev.round(1), height=250)

    # --- Tab 2: analysis ---
    with tab2:
        st.subheader("Per-cell statistics (V)")
        stats = df[cell_cols].agg(["mean", "min", "max", "std"]).T
        stats["latest"] = latest_cells
        stats["vs avg (mV)"] = (latest_cells - latest_cells.mean()) * 1000
        st.dataframe(stats.round(4), use_container_width=True)

        st.subheader("Trend & prediction")
        recent = df.tail(30)
        if len(recent) >= 5:
            x = np.arange(len(recent))
            slope_delta = np.polyfit(x, recent["v_delta"], 1)[0]        # V per sample
            slope_weak = np.polyfit(x, recent[weakest], 1)[0]           # V per sample

            t1, t2 = st.columns(2)
            t1.metric(f"{weakest} trend", f"{slope_weak * 1000:.2f} mV/sample")
            t2.metric("V_delta trend", f"{slope_delta * 1000:.2f} mV/sample")

            if slope_delta > 1e-6 and last["v_delta"] < crit_th:
                steps = (crit_th - last["v_delta"]) / slope_delta
                st.info(f"At the current trend, V_delta reaches the critical limit in about "
                        f"**{int(steps)} samples (~{int(steps * interval_sec)} sec)**.")
            elif last["v_delta"] >= crit_th:
                st.error("Critical limit already crossed.")
            else:
                st.success("V_delta is stable or improving.")
        else:
            st.caption("Collecting more samples for trend analysis...")

        st.subheader("Time spent in each state")
        states = df["v_delta"].apply(health_status).value_counts()
        st.bar_chart(states)

    # --- Tab 3: raw data ---
    with tab3:
        st.dataframe(df.sort_values("time", ascending=False), use_container_width=True, height=350)
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(),
                           file_name="battery_data.csv", mime="text/csv")


live_dashboard()