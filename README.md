# SLA Breach Predictor

An early-warning app for contact-centre supervisors. At the end of each 30-minute interval it scores the risk that the **next** interval breaches the 95% service level standard, early enough to move agents before service drops.

**Live app:** https://sla-breach-predictor-fol4yiyaod9oceshwcclkk.streamlit.app

## The problem

When I worked in workforce management (WFM) at a contact centre, we mostly saw SLA breaches in reports after they had happened. Supervisors need a warning while there is still time to act: pull agents off non-phone work, cut after-call work, or open overflow routing.

## Data

`data/simulated_call_centre.csv` holds **51,708 simulated calls** from one contact centre across 2021 (08:00 to 18:00). Each record has the call's start, answer and end times, its wait and handle time, and whether it met the service standard. The data is simulated, not real.

`simulate_data.py` groups the calls into **4,958 thirty-minute intervals** and builds the features:

| Group | Features |
|---|---|
| Volume and service | calls, SLA %, breach flag, average wait, average handle time |
| Workload | agents needed (calls × handle time ÷ 1,800 s), change in agents needed |
| Trend | change in call volume and in queue time vs the previous interval |
| Queue at the end of the interval | longest wait, calls in the last 10 minutes, their average wait, share that missed the standard |
| Time | hour, day of week, Monday and weekend flags |

The target is whether the **next interval on the same day** breaches. The last interval of each day has no target, so no label crosses overnight.

## Method

**1. Time-based split.** Neighbouring intervals are similar, so a random split leaks information and flatters the model. The data is split in time order:
- train: the earliest 60% (Jan to Aug 2021)
- validation: the next 20% (Aug to Oct)
- test: the last 20% (Oct to Dec)

**2. Model comparison.** Seven candidates were trained on the train period and compared on validation ROC AUC. ROC AUC doesn't depend on the breach rate, which rises from 14% of intervals in training to 46% in the test period.

| Model | Validation AUC | Test AUC |
|---|---|---|
| **Logistic regression (C=0.1), selected** | **0.627** | **0.646** |
| Logistic regression (C=1.0) | 0.622 | 0.636 |
| Random forest (depth 6) | 0.621 | 0.658 |
| Random forest (depth 10) | 0.614 | 0.655 |
| Gradient boosting (sklearn) | 0.613 | 0.635 |
| XGBoost (depth 3) | 0.610 | 0.630 |
| XGBoost (depth 5) | 0.591 | 0.612 |

The model was chosen on validation, not test. Random forest scores slightly higher on test, but picking it for that reason would be cherry-picking.

**3. Alert thresholds**, both chosen on validation:
- **Catch more breaches** (0.44): the highest threshold that still catches at least 80% of breaches on validation.
- **Fewer false alarms** (0.60): alerts about as often as breaches actually happen.

**4. Explanations.** The app's chart shows permutation importance: how much the model's AUC drops when each input is shuffled. The strongest drivers are workload (agents needed), calls in the last 10 minutes and the longest wait.

## Results on the test period (20 Oct to 31 Dec 2021, never seen in training)

| Setting | Breaches caught | Alerts that were real breaches | Intervals with an alert |
|---|---|---|---|
| Catch more breaches (0.44) | **87%** (398 of 458) | 49% | 82% |
| Fewer false alarms (0.60) | **57%** (261 of 458) | 57% | 46% |
| Naive rule: "it breached this interval, so warn" | 52% (238 of 458) | 52% | 46% |

**At the same number of alerts as the naive rule, the model catches more breaches (57% vs 52%) and is right more often (57% vs 52%).** AUC is 0.65 against 0.56 for the naive rule.

## Limitations

- **The data is simulated.** Results on real contact-centre data would differ.
- **The signal is modest** (AUC about 0.65). Treat the score as an early nudge to check the queue, not a certainty.
- **Breaches rise through the year,** so the 80% recall setting chosen on validation alerts on most intervals in the busier test period. Retraining regularly on recent data would help.
- **The app's inputs are typed in by hand.** A production version would read them from the ACD or telephony system every 30 minutes.

## Run it locally

```bash
pip install -r requirements.txt
python simulate_data.py     # rebuild data/processed_windows.csv
python train_model.py       # compare models, write model.joblib and model_meta.json
streamlit run app.py
```

## Files

| File | What it does |
|---|---|
| `simulate_data.py` | Turns the simulated calls into 30-minute intervals with features and the next-interval target |
| `train_model.py` | Time-based split, compares 7 models, picks thresholds, saves the winner and its report |
| `app.py` | Streamlit app: inputs, risk score, two alert settings, recommended actions, explanations, test results |
| `model.joblib` | The selected model (logistic regression pipeline) |
| `model_meta.json` | Comparison table, thresholds, test results and feature importance |

Built with Python, pandas, scikit-learn, XGBoost and Streamlit.
