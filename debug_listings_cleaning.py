import pandas as pd


df = pd.read_csv('data/listings.csv', dtype=str)
df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
df = df[df['last_review'].dt.year == 2025].copy()
print('2025 rows', len(df))
print('rows with any missing', df.isna().any(axis=1).sum())
print('rows with any empty string', (df == '').any(axis=1).sum())
print('rows with any missing or empty', ((df.isna()) | (df == '')).any(axis=1).sum())
print('rows with all values present', len(df) - ((df.isna()) | (df == '')).any(axis=1).sum())
print('columns with missing values in 2025:')
print(df.isna().sum())
print((df == '').sum())
print('\nSample of first rows with all fields present:')
complete = df.loc[~((df.isna()) | (df == '')).any(axis=1)]
print(len(complete))
print(complete.head().to_string(index=False))
