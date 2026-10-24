import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SLA Breach Predictor", layout="wide")


@st.cache_resource
def load_model():
    # The model that won the comparison in train_model.py, plus its thresholds and test results
    with open('model_meta.json', encoding='utf-8') as f:
        return joblib.load('model.joblib'), json.load(f)


model, meta = load_model()

LABELS = {'total_calls': 'Call Volume', 'avg_wait': 'Queue Time', 'avg_handle': 'Avg Handle Time', 'sla_pct': 'SLA %',
          'breach': 'Last Interval Breach', 'agents_needed': 'Agents Needed', 'staffing_gap': 'Change in Agents Needed',
          'call_volume_change': 'Volume Spike', 'wait_time_change': 'Queue Time Change', 'hour': 'Hour',
          'day_of_week': 'Day of Week', 'is_monday': 'Is Monday', 'is_weekend': 'Is Weekend',
          'max_wait': 'Longest Wait', 'calls_last10': 'Calls Last 10 Min', 'wait_last10': 'Wait Last 10 Min',
          'missed_last10': 'Missed Target Last 10 Min'}

st.title("🚨 SLA Breach Predictor")
st.subheader("Contact Centre Early Warning System")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Current Interval Data")

    # Defaults are a typical interval that met the standard (medians from the data)
    st.markdown("**Volume**")
    total_calls = st.slider("Calls This Interval", 1, 30, 9)
    call_volume_change = st.slider("Volume Change vs Last Interval (%)", -100, 250, 0) / 100

    st.markdown("**Performance**")
    avg_wait = st.slider("Average Queue Time (seconds)", 0, 300, 0)
    wait_time_change = st.slider("Queue Time Change vs Last Interval (%)", -100, 500, 0) / 100
    avg_handle = st.slider("Average Handle Time (seconds)", 10, 900, 275)
    sla_pct = st.slider("Current SLA %", 0.0, 100.0, 100.0)
    breach = 1 if sla_pct < 95 else 0
    st.caption(f"This interval {'breached' if breach else 'met'} the 95% standard.")

    st.markdown("**Staffing**")
    agents_needed = total_calls * avg_handle / 1800  # workload in agents for a 30-minute interval, as in the data
    st.caption(f"Agents needed for this workload: {agents_needed:.1f}")
    staffing_gap = st.slider("Change in Agents Needed vs Last Interval", -5.0, 5.0, 0.0, 0.1)

    st.markdown("**Queue right now (end of the interval)**")
    max_wait = st.slider("Longest Wait This Interval (seconds)", 0, 600, 0)
    calls_last10 = st.slider("Calls in the Last 10 Minutes", 0, 15, 3)
    wait_last10 = st.slider("Average Wait in the Last 10 Minutes (seconds)", 0, 300, 0)
    missed_last10 = st.slider("Calls Missing Target in the Last 10 Minutes (%)", 0, 100, 0) / 100

    st.markdown("**Time**")
    hour = st.slider("Hour of Day (interval start)", 8, 17, 12)
    day_of_week = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
    is_monday = 1 if day_of_week == 0 else 0
    is_weekend = 1 if day_of_week >= 5 else 0

    st.markdown("**Alert setting**")
    sensitivity = st.radio("When should the tool warn you?", [
        f"Fewer false alarms (threshold {meta['quiet_threshold']:.2f})",
        f"Catch more breaches (threshold {meta['threshold']:.2f})"])
    threshold = meta['threshold'] if sensitivity.startswith("Catch") else meta['quiet_threshold']

with col2:
    values = {'total_calls': total_calls, 'avg_wait': avg_wait, 'avg_handle': avg_handle, 'sla_pct': sla_pct, 'breach': breach,
              'agents_needed': agents_needed, 'staffing_gap': staffing_gap, 'call_volume_change': call_volume_change,
              'wait_time_change': wait_time_change, 'hour': hour, 'day_of_week': day_of_week, 'is_monday': is_monday,
              'is_weekend': is_weekend, 'max_wait': max_wait, 'calls_last10': calls_last10, 'wait_last10': wait_last10,
              'missed_last10': missed_last10}
    input_data = pd.DataFrame([[values[f] for f in meta['features']]], columns=meta['features'])

    probability = model.predict_proba(input_data)[0][1]
    prediction = int(probability >= threshold)

    st.subheader("🔮 Prediction")
    if prediction == 1:
        st.error(f"⚠️ SLA BREACH LIKELY in the next 30 minutes — Risk: {probability:.0%}")
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
    st.subheader("🔍 What drives the risk score?")
    importance_df = pd.DataFrame({'Feature': [LABELS[f] for f in meta['features']],
                                  'Importance': [meta['importance'][f] for f in meta['features']]}).sort_values('Importance')
    st.bar_chart(importance_df.set_index('Feature')['Importance'])
    st.caption(f"Permutation importance for the {meta['best_model']} model: how much the model's accuracy (AUC) drops "
               "when each input is shuffled. Higher = stronger influence.")

st.markdown("---")
st.markdown("### 💡 How to use this tool")
st.info("""
**This tool estimates the risk that your contact centre breaches its 95% service level standard in the next 30 minutes.**

1. **Enter the current interval's data** on the left: call volume, queue times, SLA % and the queue state in the last 10 minutes.
2. **Choose an alert setting:** "Catch more breaches" warns more often; "Fewer false alarms" warns about as often as breaches really happen.
3. **Read the risk score** and check the chart to see which inputs drive it.
4. **Follow the recommended actions** if a breach is likely. These are standard WFM interventions.

*Trained on a simulated contact-centre dataset of 51,708 calls (2021), grouped into 30-minute intervals.*
""")

t, q, b = meta['test'], meta['test_quiet'], meta['baseline_last_interval_breached']
with st.expander("📈 How well does it work, and which model is this?"):
    rows = "\n".join(f"| {r['model']} | {r['val_auc']:.3f} | {r['test_auc']:.3f} |" for r in meta['comparison'])
    st.markdown(f"""
**Model:** {meta['best_model']}, chosen from {len(meta['comparison'])} candidates by validation AUC (time-based split:
train on the earliest 60% of intervals, choose on the next 20%, test once on the last 20%).

| Model | Validation AUC | Test AUC |
|---|---|---|
{rows}

**Test period** ({meta['periods']['test'][0][:10]} to {meta['periods']['test'][1][:10]}), which the model never saw:

| Setting | Breaches caught | Alerts that were real breaches | Intervals with an alert |
|---|---|---|---|
| Catch more breaches | {t['recall']:.0%} ({t['caught']} of {t['breaches']}) | {t['precision']:.0%} | {t['alert_share']:.0%} |
| Fewer false alarms | {q['recall']:.0%} ({q['caught']} of {q['breaches']}) | {q['precision']:.0%} | {q['alert_share']:.0%} |
| Simple rule: "it breached this interval" | {b['recall']:.0%} | {b['precision']:.0%} | {b['alert_share']:.0%} |

The data is simulated and breaches become more common through the year (about {meta['breach_rate_by_period']['train']:.0%} of
intervals in training against {meta['breach_rate_by_period']['test']:.0%} in the test period), so the signal is modest:
AUC {meta['auc']['best_test']:.2f} against {meta['auc']['baseline_test']:.2f} for the simple rule. Treat the risk score as an early
nudge to check the queue, not a certainty.
""")
