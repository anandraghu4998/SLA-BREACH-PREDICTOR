import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="SLA Breach Predictor", layout="wide")

model = joblib.load('model.pkl')

st.title("🚨 SLA Breach Predictor")
st.subheader("Contact Centre Early Warning System")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Current Interval Data")
    
    st.markdown("**Volume**")
    total_calls = st.slider("Call Volume for Last Interval", 1, 100, 20)
    call_volume_change = st.slider("Volume Spike vs Last Interval (%)", -1.0, 2.0, 0.0)
    
    st.markdown("**Staffing**")
    agents_needed = st.slider("Agents Available", 1, 50, 10)
    staffing_gap = st.slider("Agents Absent", -20, 20, 0)
    
    st.markdown("**Performance**")
    avg_wait = st.slider("Queue Time (seconds)", 0, 300, 30)
    wait_time_change = st.slider("Queue Time Change vs Last Interval (%)", -1.0, 2.0, 0.0)
    avg_handle = st.slider("Average Handle Time (seconds)", 60, 1200, 400)
    sla_pct = st.slider("Current SLA %", 0.0, 100.0, 95.0)
    breach = st.selectbox("Did Last Interval Breach SLA?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    st.markdown("**Time**")
    hour = st.slider("Hour of Day", 0, 23, 9)
    day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 0)
    is_monday = 1 if day_of_week == 0 else 0
    is_weekend = 1 if day_of_week >= 5 else 0

with col2:
    input_data = pd.DataFrame([[
        total_calls, avg_wait, avg_handle, sla_pct, breach,
        agents_needed, staffing_gap, call_volume_change, wait_time_change,
        hour, day_of_week, is_monday, is_weekend
    ]], columns=[
        'total_calls', 'avg_wait', 'avg_handle', 'sla_pct', 'breach',
        'agents_needed', 'staffing_gap', 'call_volume_change', 'wait_time_change',
        'hour', 'day_of_week', 'is_monday', 'is_weekend'
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("🔮 Prediction")
    if prediction == 1:
        st.error(f"⚠️ SLA BREACH PREDICTED in next 30 minutes — Risk: {probability:.0%}")
        st.markdown("### 🛠️ Recommended Actions")
        st.markdown("""
        - Pull agents from non-critical tasks immediately
        - Reduce after-call work time if possible
        - Escalate to operations manager
        - Open overflow routing if available
        - Check for tech issues causing high handle time
        """)
    else:
        st.success(f"✅ SLA on track — Breach Risk: {probability:.0%}")
        st.markdown("### 👍 All Good")
        st.markdown("""
        - Continue monitoring every 30 minutes
        - Watch for sudden volume spikes
        - Ensure scheduled agents are on the floor
        """)

    st.markdown("---")
    st.subheader("🔍 Why this prediction?")

    importance = model.feature_importances_
    feature_names = ['Call Volume', 'Queue Time', 'Avg Handle Time', 'SLA %', 'Last Interval Breach',
                    'Agents Available', 'Agents Absent', 'Volume Spike', 'Queue Time Change',
                    'Hour', 'Day of Week', 'Is Monday', 'Is Weekend']

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)

    st.bar_chart(importance_df.set_index('Feature')['Importance'])
    st.caption("Higher value = stronger influence on the prediction.")

st.markdown("---")
st.markdown("### 💡 How to use this tool")
st.info("""
**This tool predicts whether your contact centre will breach its SLA target in the next 30 minutes.**

1. **Enter your current interval data** on the left — call volume, agents available, queue time, and current SLA %.
2. **The prediction updates instantly** — green means you're on track, red means a breach is likely in the next interval.
3. **Check the chart** to understand why — it shows which factors have the strongest influence on the prediction.
4. **Follow the recommended actions** if a breach is predicted — these are standard WFM interventions used in real contact centres.

*Built using an XGBoost ML model trained on 51,000 real contact centre call records.*
""")