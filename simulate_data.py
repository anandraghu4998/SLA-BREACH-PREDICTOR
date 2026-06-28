import pandas as pd
import numpy as np

df=pd.read_csv('data/simulated_call_centre.csv')
print(df.shape)
print(df.head())
print(df.dtypes)


df['date']=pd.to_datetime(df['date'])
df['call_started']=pd.to_datetime(df['call_started'],format='%I:%M:%S %p')

df['hour']=df['call_started'].dt.hour
df['day_of_week']=df['date'].dt.dayofweek
df['month']=df['date'].dt.month

print(df[['date','hour','day_of_week','month']].head())

df['datetime']=pd.to_datetime(df['date'].astype(str)+' '+df['call_started'].dt.strftime('%H:%M:%S'))
df=df.sort_values('datetime')
df['windows']=df['datetime'].dt.floor('30min')

windows_stats= df.groupby('windows').agg(
    total_calls=('call_id','count'),
    met_standard=('meets_standard','sum'),
    avg_wait=('wait_length','mean'),
    avg_handle=('service_length','mean')
).reset_index()

windows_stats['sla_pct']=windows_stats['met_standard']/windows_stats['total_calls']*100
windows_stats['breach']=(windows_stats['sla_pct']<95).astype(int)

print(windows_stats.head(10))
print(f"\nBreach windows: {windows_stats['breach'].sum()}")
print(f"Total windows:{len(windows_stats)}")

windows_stats['next_breach'] = windows_stats['breach'].shift(-1)
windows_stats = windows_stats.dropna()
windows_stats['next_breach'] = windows_stats['next_breach'].astype(int)

windows_stats['calls_per_agent'] = windows_stats['avg_handle'] / 1800
windows_stats['agents_needed'] = windows_stats['total_calls'] * windows_stats['calls_per_agent']
windows_stats['staffing_gap'] = windows_stats['agents_needed'] - windows_stats['agents_needed'].shift(1)

windows_stats['call_volume_change'] = windows_stats['total_calls'].pct_change()
windows_stats['wait_time_change'] = windows_stats['avg_wait'].pct_change()
windows_stats = windows_stats.replace([np.inf, -np.inf], np.nan)
windows_stats['staffing_gap'] = windows_stats['staffing_gap'].fillna(0)
windows_stats['call_volume_change'] = windows_stats['call_volume_change'].fillna(0)
windows_stats['wait_time_change'] = windows_stats['wait_time_change'].fillna(0)
windows_stats = windows_stats.dropna()

windows_stats['hour'] = windows_stats['windows'].dt.hour
windows_stats['day_of_week'] = windows_stats['windows'].dt.dayofweek
windows_stats['is_monday'] = (windows_stats['day_of_week'] == 0).astype(int)
windows_stats['is_weekend'] = (windows_stats['day_of_week'] >= 5).astype(int)

print(windows_stats[['windows', 'sla_pct', 'breach', 'next_breach']].head(10))
windows_stats.to_csv('data/processed_windows.csv', index=False)
print("Saved processed data.")