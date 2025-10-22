import pandas as pd

# Lag values
def make_lags(y, lags):
    try:
        df = pd.DataFrame({'y': y})
    except:
        df = pd.DataFrame({'y': y.iloc[:, 0]})
    for i in range(1, lags + 1):
        df[f'lag{i}'] = df['y'].shift(i)
    df = df.drop(columns=['y'])
    return df.dropna()

# Shock values
def make_shocks(y, mean):
    try:
        df = pd.DataFrame({'y': y})
    except:
        df = pd.DataFrame({'y': y.iloc[:, 0]})
    df['shock'] = df['y'] - mean
    df = df.drop(columns=['y'])
    return df.dropna()