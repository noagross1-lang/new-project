import math
import numpy as np
import pandas as pd

df = pd.read_csv('data/listings_model_predictions.csv')

# Keep rows with price and prediction
df = df.dropna(subset=['predicted_price', 'price_clean', 'log_price', 'predicted_log_price'])

n = len(df)
mean_price = df['price_clean'].mean()
median_price = df['price_clean'].median()

rmse_price = math.sqrt(((df['price_clean'] - df['predicted_price']) ** 2).mean())
mae_price = (df['price_clean'] - df['predicted_price']).abs().mean()
rmse_log = math.sqrt(((df['log_price'] - df['predicted_log_price']) ** 2).mean())

# MAPE (ignore zeros)
valid = df['price_clean'].replace(0, np.nan).dropna()
if len(valid) > 0:
    mape = ( (df.loc[valid.index, 'price_clean'] - df.loc[valid.index, 'predicted_price']).abs() / df.loc[valid.index, 'price_clean'] ).mean() * 100
    medape = ( (df.loc[valid.index, 'price_clean'] - df.loc[valid.index, 'predicted_price']).abs() / df.loc[valid.index, 'price_clean'] ).median() * 100
else:
    mape = float('nan')
    medape = float('nan')

mult_factor = math.exp(rmse_log)

print(f"n={n}")
print(f"mean_price={mean_price:.2f}, median_price={median_price:.2f}")
print(f"RMSE(price)={rmse_price:.2f}")
print(f"MAE(price)={mae_price:.2f}")
print(f"RMSE(log_price)={rmse_log:.4f}")
print(f"Multiplicative error ~ e^RMSE_log = {mult_factor:.3f}x")
print(f"MAPE={mape:.2f}%, Median APE={medape:.2f}%")
