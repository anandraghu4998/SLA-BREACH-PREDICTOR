"""Turn the simulated call-level data into 30-minute intervals for the model.

Input:  data/simulated_call_centre.csv (51,708 simulated calls, Jan to Dec 2021)
Output: data/processed_windows.csv (one row per 30-minute interval)

Each interval gets volume, service level and staffing features, plus the queue state at the end of the interval
(the last 10 minutes), which is what a WFM analyst watches in real time. The target `next_breach` is whether the
NEXT interval on the SAME day breaches the 95% standard. The last interval of each day has no target and is
dropped, so no label crosses overnight.
"""

import numpy as np
import pandas as pd

df = pd.read_csv('data/simulated_call_centre.csv')
print(df.shape)

df['date'] = pd.to_datetime(df['date'])
started = pd.to_datetime(df['call_started'], format='%I:%M:%S %p')
df['datetime'] = df['date'] + pd.to_timedelta(started.dt.hour * 3600 + started.dt.minute * 60 + started.dt.second, unit='s')
df = df.sort_values('datetime')
df['windows'] = df['datetime'].dt.floor('30min')
df['met'] = df['meets_standard'].astype(str).str.upper().eq('TRUE')
df['last10'] = df['datetime'] >= df['windows'] + pd.Timedelta(minutes=20)

windows_stats = df.groupby('windows').agg(
    total_calls=('call_id', 'count'),
    met_standard=('met', 'sum'),
    avg_wait=('wait_length', 'mean'),
    avg_handle=('service_length', 'mean'),
    max_wait=('wait_length', 'max'),
).reset_index()

# Queue state in the last 10 minutes of the interval
last10 = df[df['last10']].groupby('windows').agg(
    calls_last10=('call_id', 'count'),
    wait_last10=('wait_length', 'mean'),
    met_last10=('met', 'mean'),
).reset_index()
windows_stats = windows_stats.merge(last10, on='windows', how='left')
windows_stats['missed_last10'] = 1 - windows_stats.pop('met_last10')
windows_stats[['calls_last10', 'wait_last10', 'missed_last10']] = windows_stats[['calls_last10', 'wait_last10', 'missed_last10']].fillna(0)

windows_stats['sla_pct'] = windows_stats['met_standard'] / windows_stats['total_calls'] * 100
windows_stats['breach'] = (windows_stats['sla_pct'] < 95).astype(int)
print(f"Breach intervals: {windows_stats['breach'].sum()} of {len(windows_stats)}")

by_day = windows_stats.groupby(windows_stats['windows'].dt.date)
windows_stats['next_breach'] = by_day['breach'].shift(-1)          # same day only

windows_stats['calls_per_agent'] = windows_stats['avg_handle'] / 1800
windows_stats['agents_needed'] = windows_stats['total_calls'] * windows_stats['calls_per_agent']
by_day = windows_stats.groupby(windows_stats['windows'].dt.date)
windows_stats['staffing_gap'] = (windows_stats['agents_needed'] - by_day['agents_needed'].shift(1)).fillna(0)
windows_stats['call_volume_change'] = by_day['total_calls'].pct_change()
windows_stats['wait_time_change'] = by_day['avg_wait'].pct_change()
windows_stats = windows_stats.replace([np.inf, -np.inf], np.nan)
windows_stats[['call_volume_change', 'wait_time_change']] = windows_stats[['call_volume_change', 'wait_time_change']].fillna(0)

windows_stats['hour'] = windows_stats['windows'].dt.hour
windows_stats['day_of_week'] = windows_stats['windows'].dt.dayofweek
windows_stats['is_monday'] = (windows_stats['day_of_week'] == 0).astype(int)
windows_stats['is_weekend'] = (windows_stats['day_of_week'] >= 5).astype(int)

windows_stats = windows_stats.dropna(subset=['next_breach'])
windows_stats['next_breach'] = windows_stats['next_breach'].astype(int)

print(windows_stats[['windows', 'sla_pct', 'breach', 'next_breach', 'wait_last10']].head(10))
windows_stats.to_csv('data/processed_windows.csv', index=False)
print(f"Saved {len(windows_stats)} intervals to data/processed_windows.csv")
