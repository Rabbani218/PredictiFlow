import pathlib
import sys
import pandas as pd

root = pathlib.Path(__file__).resolve().parents[1]
data_path = root / 'data' / 'hypercar_sales.csv'
if not data_path.exists():
    print('hypercar data not found:', data_path)
    sys.exit(2)

df = pd.read_csv(data_path)
print('Loaded hypercar data rows:', len(df))
print('Columns:', df.columns.tolist())
print('\nUnique models:', df['model'].unique().tolist())

print('\nCounts per model:')
print(df.groupby('model').size())

print('\nSample rows per model:')
for m, g in df.groupby('model'):
    print('---', m, 'rows:', len(g))
    print(g.head(2).to_string(index=False))

print('\nDone demo.')
