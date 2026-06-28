import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import shap

df= pd.read_csv('data/processed_windows.csv')
print(df.shape)
print(df.head())

features = ['total_calls', 'avg_wait', 'avg_handle', 'sla_pct', 'breach',
            'agents_needed', 'staffing_gap', 'call_volume_change', 'wait_time_change',
            'hour', 'day_of_week', 'is_monday', 'is_weekend']
target='next_breach'

X=df[features]
y=df[target]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

model=XGBClassifier(n_estimators=200, max_depth=4, random_state=42,scale_pos_weight=2.91)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))
print(f"Breach rate: {y.mean():.2%}")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=features)

import joblib

joblib.dump(model, 'model.pkl')
print("Model saved.")