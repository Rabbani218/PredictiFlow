from backend.app.core.forecasting import _detect_value_column, prepare_df
import pandas as pd
p='data/harga_laptop.csv'
df=pd.read_csv(p)
print('columns:', list(df.columns))
print('dtype sample:')
print(df.dtypes)
vc=_detect_value_column(df)
print('detected value col:', vc)
# Also try date detection logic by calling prepare_df but catching error to get message
try:
    _ = prepare_df(df)
    print('prepare_df succeeded; sample head:')
    print(_ .head())
except Exception as e:
    print('prepare_df error:', e)
