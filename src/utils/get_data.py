from vnstock import Quote
import pandas as pd
import numpy as np

# Get risk free
def get_risk_free(FREE_RISK_PATH):
    df = pd.read_excel(FREE_RISK_PATH)

    return max(df['rf'].mean(), 0)

# Get data
def get_stock(stock_name, start_date, end_date):
    quote = Quote(symbol=stock_name)
    df = quote.history( start=start_date,
                        end=end_date,
                        resolution='1D')
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close']).diff()
    df.drop(0, inplace=True)
    return df