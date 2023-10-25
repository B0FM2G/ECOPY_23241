import pandas as pd
data = pd.read_parquet('../../data/sp500.parquet', engine='fastparquet')

data2 = pd.read_parquet('../../data/ff_factors.parquet', engine='fastparquet')

merged = data.merge(data2, on='Date', how='left')

merged['Excess Return'] = merged['Monthly Returns'] - merged['RF']

merged = merged.sort_index()
merged['ex_ret_1'] = merged.groupby('Symbol')['Excess Return'].shift(-1)

merged = merged.dropna(subset=['ex_ret_1'])
merged = merged.dropna(subset=['HML'])

merged = merged[merged['Symbol'] == 'AMZN']
merged = merged.drop(columns=['Symbol'])
