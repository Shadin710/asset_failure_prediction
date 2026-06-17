import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Page Config ---
st.set_page_config(page_title="Predictive Maintenance Tool", layout="wide")
st.title("⚙️ Engineering Asset Diagnostic Dashboard")

# --- Load Models ---
@st.cache_resource
def load_models():
    with open("models/tier1_binary_model.pkl", "rb") as f1, open("models/tier2_diagnostic_model.pkl", "rb") as f2:
        return pickle.load(f1), pickle.load(f2)

binary_model, diagnostic_model = load_models()

# --- Sidebar Inputs (The Interactive Sliders) ---
st.sidebar.header("Machine Telemetry Inputs")
air_temp = st.sidebar.slider("Air Temperature [K]", 290.0, 310.0, 298.1)
process_temp = st.sidebar.slider("Process Temperature [K]", 300.0, 320.0, 308.0)
rpm = st.sidebar.slider("Rotational Speed [rpm]", 1000, 2500, 1551)
torque = st.sidebar.slider("Torque [Nm]", 10.0, 80.0, 42.8)
tool_wear = st.sidebar.slider("Tool Wear [min]", 0, 250, 0)
machine_type = st.sidebar.selectbox("Machine Quality Type", ["L", "M", "H"], index=1)

# --- Feature Calculation ---
quality_mapping = {'L': 0, 'M': 1, 'H': 2}
telemetry_data = pd.DataFrame([{
    'Air temperature [K]': air_temp,
    'Process temperature [K]': process_temp,
    'Rotational speed [rpm]': rpm,
    'Torque [Nm]': torque,
    'Tool wear [min]': tool_wear,
    'Thermal_Diff': process_temp - air_temp,
    'Mechanical_Power': (2 * np.pi * rpm * torque) / 60,
    'Frictional_Wear_Index': torque * tool_wear,
    'Type_Quality': quality_mapping[machine_type]
}])

# --- Dashboard Display ---
st.subheader("Calculated Physical Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Temperature Differential", f"{telemetry_data['Thermal_Diff'][0]:.2f} K")
col2.metric("Mechanical Power", f"{telemetry_data['Mechanical_Power'][0]:.0f} W")
col3.metric("Frictional Wear Index", f"{telemetry_data['Frictional_Wear_Index'][0]:.0f}")

st.divider()

# --- Inference Engine ---
is_failing = binary_model.predict(telemetry_data)[0]

st.subheader("System Status")
if is_failing == 0:
    st.success("✅ Operational. Continue standard telemetry logging.")
else:
    failure_mode = diagnostic_model.predict(telemetry_data)[0]
    st.error("⚠️ CRITICAL RISK DETECTED")
    st.warning(f"**Diagnosed Cause:** {failure_mode}")
    
    # Prescriptions
    if failure_mode == "Tool Wear Failure":
        st.info("**Action:** Replace machining insert head immediately.")
    elif failure_mode == "Heat Dissipation Failure":
        st.info("**Action:** Clean ventilation matrix and inspect internal coolant fluid.")
    elif failure_mode == "Power Failure":
        st.info("**Action:** Drive motor load limits exceeded. Reduce load.")
    else:
        st.info("**Action:** Schedule urgent hardware diagnostics.")