# %% Imports, constants, data loading
# # # Imports, constants, data loading # # # # # # # # # # # # # # # # # # # # # # # # 
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

import numpy as np
import pandas as pd
from datetime import time

import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import shap

from features import add_time_features, add_vix_features, add_lag_features
from model import walk_forward_validation


PATH = "data/US500_m1_bars.csv"
# PATH = "data/US500_m5_bars.csv"
# PATH = "data/US500_m1_bars.csv"

# PATH_VIX_TERM_STRUCTURE = '../barchart-download/output/5m-bars-attempt-3/term_structure_5m.csv'
SHIFT = 5 # the number of bars to shift the target variable

# comment out all other TARGET_TYPE constants, function calls will adjust accordingly
TARGET_TYPE = 'regressor' # based on 'close' column

# TARGET_TYPE = 'classifier' # based on future close minus current close, produces 0, 1 or 2 for
                           # set criteria in the 'make_supervised' function
PIP_CHANGE = 8             # only used if 'classifier' TARGET_TYPE is selected

## load the dataset
df = pd.read_csv(PATH)
df = df.set_index('Time')
df.index = df.index.astype('datetime64[ns]')
df = df.drop(df.columns[0], axis=1)
df = df.sort_index()
df = df.tail(100000)

df = add_time_features(df)
# df = add_vix_features(df, PATH_VIX_TERM_STRUCTURE)
# df = add_lag_features(df)

def make_supervised(df, target_type) -> pd.DataFrame:
    """ Takes a dataframe and returns a dataframe with a target column
    @ params: dataframe
    @ returns: dataframe with new target column based on 'Close' column
    """
    
    if target_type == 'classifier':
        None # TODO: implement
        
        close_on_high = df['Close'] / df['High']
        close_on_low = df['Close'] / df['Low']
        df['target'] = ((close_on_high + close_on_low) / 2) - 1
        
        
        # df['target'] = df['Close'].shift(-SHIFT) - df['Close']
        # # df['target1'] = df['Close'].shift(-SHIFT) - df['Close']
        # df['target'] = df['target'].apply(lambda x: 2 if x > PIP_CHANGE else 
        #                                   (0 if x < -PIP_CHANGE else 1))
    
    if target_type == 'regressor':
        # Absolute value as target
        shift = df['Close'].shift(-SHIFT)
        df['target'] = shift
        # drop nan rows of target column
        df = df.dropna(subset=['target'])
    
    return df

supervised = make_supervised(df, TARGET_TYPE)
supervised.head(20)

# supervised.sort_values(by='target', ascending=False).head(20)
# %% Plot Target
# # # Plot Target # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

target_fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True, vertical_spacing = 0.01, row_heights = [0.6, 0.4])

target_fig.add_trace(go.Scatter(x=supervised.index, y=supervised['Close'],
                             mode='lines', name='Close', 
                             line=dict(color='blue', width=1)), 
                  row=1, col=1)

target_fig.add_trace(go.Scatter(x=supervised.index, y=supervised['target'],
                             mode='lines', name='Target', 
                             line=dict(color='red', width=1)), 
                  row=2, col=1)

target_fig.update_layout(height=900, title_text="Predictions")


# %% Model Training
# # # Model Training # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

supervised_np = supervised.values

TEST_SET_SIZE = 20000
print(f"Test set size: {TEST_SET_SIZE}\nShift: {SHIFT}")
error, y, yhat, model = walk_forward_validation(supervised_np, TEST_SET_SIZE, target_type=TARGET_TYPE)
print('error: %.3f' % error)


# %% Plot Predictions
# # # Plot Predictios # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def plot_importances(model):
    #sorted in reverse order
    sorted_idx = model.feature_importances_.argsort()
    features = df.columns[:-1]
    # plot the feature impotances agains the column names
    fig = go.Figure()
    fig.add_trace(go.Bar(y=features[sorted_idx], x=model.feature_importances_[sorted_idx], orientation='h'))
    fig.update_layout(height=400, title_text="Feature Importances")
    fig.show()
plot_importances(model)

def plot_predictions(plot_df):
    
    fig = make_subplots(rows = 2 if TARGET_TYPE == 'classifier' else 1, 
                        cols = 1,
                        shared_xaxes = True, 
                        vertical_spacing = 0.01, 
                        row_heights = [0.6, 0.4] if TARGET_TYPE == 'classifier' else None)
    
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Close'],
                            mode='lines',
                            name='Close', 
                            line=dict(color='blue', width=1),
                            ), row=1, col=1)
    
    if TARGET_TYPE == 'regressor':
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['preds'].shift(SHIFT),
                                mode='lines',
                                name='Preds', 
                                marker=dict(color='red', size=1),
                                line=dict(color='red', width=1),
                                ), row=1, col=1)
                                 
    if TARGET_TYPE == 'classifier':
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['preds'].shift(SHIFT),
                                mode='markers',
                                name='Prediction',
                                line=dict(color='red', width=3),
                                ), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['target'].shift(SHIFT),
                                mode='lines',
                                name='Target',
                                marker=dict(color='green', size=0.5),
                                line=dict(color='green', width=1),
                                ), row=2, col=1)
    
    fig.update_xaxes(rangebreaks=[
        dict(bounds=["sat", "sun"]), #hide weekends,
    ])
    fig.update_layout(height=900, title_text="Predictions")
    
    fig.show()
    
df_for_plot = df
preds = pd.DataFrame(yhat, index=df.index[-(TEST_SET_SIZE+SHIFT):-SHIFT], columns=['preds'])
plot = df_for_plot.join(preds, how='outer').tail(500 + TEST_SET_SIZE)

plot_predictions(plot)