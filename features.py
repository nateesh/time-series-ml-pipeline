import pandas as pd
from datetime import time


def add_time_features(df) -> pd.DataFrame:
    """ Takes a dataframe, adds time features and returns a dataframe
    @ params: csv file path
    @ returns: dataframe
    """
    
    # Establish trading hours for NYSE
    df['Hour'] = df.index.hour
    df['WeekDay'] = df.index.weekday # 0 = Monday, 1 = Tuesday, 2 = Wednesday

    pre_market_hour = ((df.index.weekday >= 0) & (df.index.weekday <= 4) & 
                    (df.index.time >= time(6,30)) & (df.index.time < time(9,30)))
    trading_hour = ((df.index.weekday >= 0) & (df.index.weekday <= 4) & 
                    (df.index.time >= time(9,30)) & (df.index.time < time(16)))

    df['isPreMarket'] = pre_market_hour
    df['isTrading'] = trading_hour
    
    return df

def add_vix_features(df, path):
    vix_ts = pd.read_csv(path, index_col=0, parse_dates=True)
    # vix_ts = vix_ts.set_index('Date')
    # vix_ts.index = vix_ts.index.astype('datetime64[ns]')
    
    # vratio is the ratio of the 3 month vix to the vix... > than 1 is 'risk On'
    vix_ts['vratio'] = vix_ts['VX3'] / vix_ts['C']
    
    # contango is the ratio of the 2 month vix to the 1 month vix... > than -0.05 is 'risk-on'
    vix_ts['contango'] = vix_ts['VX2'] / vix_ts['VX1'] - 1
    
    # contango roll is the ratio of the 2 month vix to the vix minus 1... > than 0.1 is 'risk-on'
    vix_ts['contango_roll'] = vix_ts['VX2'] / vix_ts['C'] - 1

    df = df.join(vix_ts, how='left')
    
    return df

def add_lag_features(df):
    """
    Add lag features based off of previous OHLCV data
    """
    df['close_minus_open'] = df['Close'] - df['Open']

    df['close_on_low'] = df['Close'] / df['Low']
    df['close_on_high'] = df['Close'] / df['High']

    # create rolling mean, first shift by 1 to avoid leakage
    for roll in [3, 4, 5, 6, 10, 15, 30]:
        df[f'close_rolling_mean_{roll}'] = df['Close'].shift(-1).rolling(roll).mean()
        df[f'high_rolling_mean_{roll}'] = df['High'].shift(-1).rolling(roll).mean()
        df[f'low_rolling_mean_{roll}'] = df['Low'].shift(-1).rolling(roll).mean()
  
    # create the lag features but keep them in order
    LAG = 12
    for i in range(1, LAG):
        df[f'close_on_low_lag{i}'] = df['Close'].shift(i) / df['Low'].shift(i)
        df[f'close_on_high_lag{i}'] = df['Close'].shift(i) / df['High'].shift(i)
        df[f'HL_range_lag_{i}'] = df['Low'].shift(i)
        df[f'close_lag_{i}'] = df['Close'].shift(i)
        # df[f'open_lag_{i}'] = df['Open'].shift(i)
        # df[f'high_lag_{i}'] = df['High'].shift(i)
        # df[f'low_lag_{i}'] = df['Low'].shift(i)
        # df[f'vol_lag_{i}'] = df['Volume'].shift(i)
    
    return df

def make_supervised(df) -> pd.DataFrame:
    """ Takes a dataframe and returns a dataframe with a target column
    @ params: dataframe
    @ returns: dataframe with new target column based on 'Close' column
    """
    
    if TARGET_TYPE == 'classifier':
        None # TODO: add classification target
        
    if TARGET_TYPE == 'regressor':
        # Absolute value as target
        shift = df['Close'].shift(-SHIFT)
        df['target'] = shift
        # drop nan rows of target column
        df = df.dropna(subset=['target'])
    
    return df