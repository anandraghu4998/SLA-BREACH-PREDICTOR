"""Compare models for SLA breach early warning and save the best one.

Data: data/processed_windows.csv, the 30-minute intervals built by simulate_data.py from the simulated
contact-centre dataset (data/simulated_call_centre.csv, 51,708 calls in 2021).

Evaluation is time-based, because neighbouring intervals are similar and a random split leaks:
  - train      = the earliest 60% of intervals (models are fitted here)
  - validation = the next 20% (used to pick the model, its settings and the alert threshold)
  - test       = the latest 20% (scored once, for the final report)

Model choice: the candidate with the highest validation ROC AUC. AUC does not depend on the breach rate, which
rises through the year in this data (about 14% in train against 46% in test).
Alert threshold: the highest threshold that still catches TARGET_RECALL of breaches on validation.

Outputs: model.joblib (the winning fitted model) and model_meta.json (comparison table, threshold, test results,
feature importance), both read by app.py.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

TARGET_RECALL = 0.80
FEATURES = ['total_calls', 'avg_wait', 'avg_handle', 'sla_pct', 'breach',
            'agents_needed', 'staffing_gap', 'call_volume_change', 'wait_time_change',
            'hour', 'day_of_week', 'is_monday', 'is_weekend',
            'max_wait', 'calls_last10', 'wait_last10', 'missed_last10']  # queue state at the end of the interval
TARGET = 'next_breach'

df = pd.read_csv('data/processed_windows.csv', parse_dates=['windows']).sort_values('windows').reset_index(drop=True)
n = len(df)
train, val, test = df.iloc[:int(n * 0.6)], df.iloc[int(n * 0.6):int(n * 0.8)], df.iloc[int(n * 0.8):]
Xtr, ytr, Xva, yva, Xte, yte = train[FEATURES], train[TARGET], val[FEATURES], val[TARGET], test[FEATURES], test[TARGET]
pos_weight = (ytr == 0).sum() / (ytr == 1).sum()

CANDIDATES = {
    'Logistic regression (C=1.0)': make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')),
    'Logistic regression (C=0.1)': make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight='balanced')),
    'Random forest (depth 6)': RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=20, class_weight='balanced', random_state=42, n_jobs=-1),
    'Random forest (depth 10)': RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=10, class_weight='balanced', random_state=42, n_jobs=-1),
    'Gradient boosting (sklearn)': HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, class_weight='balanced', random_state=42),
    'XGBoost (depth 3)': XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, random_state=42, scale_pos_weight=pos_weight),
    'XGBoost (depth 5)': XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8, random_state=42, scale_pos_weight=pos_weight),
}


def scores(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'recall': round(recall_score(y, pred), 3), 'precision': round(precision_score(y, pred, zero_division=0), 3),
            'f1': round(f1_score(y, pred), 3), 'breaches': int(tp + fn), 'caught': int(tp), 'alerts': int(tp + fp),
            'alert_share': round((tp + fp) / len(y), 3)}


comparison, fitted = [], {}
for name, model in CANDIDATES.items():
    model.fit(Xtr, ytr)
    pv, pt = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
    fitted[name] = model
    comparison.append({'model': name, 'val_auc': round(roc_auc_score(yva, pv), 3), 'val_pr_auc': round(average_precision_score(yva, pv), 3),
                       'test_auc': round(roc_auc_score(yte, pt), 3), 'test_pr_auc': round(average_precision_score(yte, pt), 3)})
comparison.sort(key=lambda r: (r['val_auc'], r['val_pr_auc']), reverse=True)
best_name = comparison[0]['model']
best = fitted[best_name]

# Alert threshold, chosen on validation only.
val_proba, test_proba = best.predict_proba(Xva)[:, 1], best.predict_proba(Xte)[:, 1]
grid = np.round(np.arange(0.01, 0.96, 0.01), 2)
ok = [t for t in grid if recall_score(yva, (val_proba >= t).astype(int)) >= TARGET_RECALL]
threshold = float(max(ok)) if ok else float(grid[0])
# Second setting, "fewer false alarms": alert about as often as breaches actually happen on validation.
quiet = float(grid[np.argmin([abs((val_proba >= t).mean() - yva.mean()) for t in grid])])

# Global feature importance that works for any model type: permutation importance on validation (AUC drop).
perm = permutation_importance(best, Xva, yva, scoring='roc_auc', n_repeats=10, random_state=42)
importance = {f: round(float(v), 4) for f, v in zip(FEATURES, perm.importances_mean)}

baseline = test['breach'].astype(int)  # naive rule: "it breached this interval, so warn about the next one"
meta = {
    'best_model': best_name,
    'selection': 'highest validation ROC AUC (tie-break: validation PR AUC)',
    'comparison': comparison,
    'threshold': threshold,
    'target_recall': TARGET_RECALL,
    'periods': {name: [str(part['windows'].min()), str(part['windows'].max()), len(part)]
                for name, part in (('train', train), ('validation', val), ('test', test))},
    'breach_rate_by_period': {name: round(float(part[TARGET].mean()), 3) for name, part in (('train', train), ('validation', val), ('test', test))},
    'validation': scores(yva, (val_proba >= threshold).astype(int)),
    'test': scores(yte, (test_proba >= threshold).astype(int)),
    'quiet_threshold': quiet,
    'test_quiet': scores(yte, (test_proba >= quiet).astype(int)),
    'test_at_default_0.5': scores(yte, (test_proba >= 0.5).astype(int)),
    'baseline_last_interval_breached': scores(yte, baseline),
    'auc': {'best_test': round(roc_auc_score(yte, test_proba), 3), 'baseline_test': round(roc_auc_score(yte, baseline), 3)},
    'importance': importance,
    'features': FEATURES,
}

joblib.dump(best, 'model.joblib')
with open('model_meta.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2)

print(f"Intervals: {n} (train {len(train)}, validation {len(val)}, test {len(test)}); breach rate by period {meta['breach_rate_by_period']}")
print(f"\n{'model':30s} {'val AUC':>8s} {'val PR':>7s} {'test AUC':>9s} {'test PR':>8s}")
for r in comparison:
    print(f"{r['model']:30s} {r['val_auc']:8.3f} {r['val_pr_auc']:7.3f} {r['test_auc']:9.3f} {r['test_pr_auc']:8.3f}")
print(f"\nBest on validation: {best_name}; alert thresholds: catch more {threshold}, fewer false alarms {quiet}")
for name in ('validation', 'test', 'test_quiet', 'test_at_default_0.5', 'baseline_last_interval_breached'):
    s = meta[name]
    print(f"{name:34s} caught {s['caught']}/{s['breaches']} (recall {s['recall']:.1%}), precision {s['precision']:.1%}, alerts on {s['alert_share']:.0%} of intervals")
print(f"AUC: {meta['auc']}")
top = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:5]
print("Top features (permutation, AUC drop):", ", ".join(f"{k} {v:.3f}" for k, v in top))
print("Saved model.joblib and model_meta.json")
