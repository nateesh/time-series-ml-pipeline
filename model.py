import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor, XGBClassifier

def train_test_split(data, n_test):
    """ Splits data into train and test sets
    @ params: dataframe supervised dataset
    @ params: n_test: number of rows to use for test set
    @ returns: train, test sets as tuple of numpy arrays """
    return data[:-n_test, :], data[-n_test:, :]

def xgboost_forecast(train, target_type) -> np.array:
    # transform list into array
    train = np.asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    if target_type == 'classifier':
        None # TODO
    
    if target_type == 'regressor':
        model = XGBRegressor(objective='reg:squarederror',
                             n_estimators=100,
                             tree_method='gpu_hist', gpu_id=0,
                             sampling_method='uniform',
                             max_bin=623,
                             max_depth=12,
                             alpha=0.5108154566815425,
                             gamma=1.9276236172849432,
                             reg_lambda=11.40999855634382,
                             colsample_bytree=0.705851334291963,
                             subsample=0.8386116751473301,
                             min_child_weight=2.5517043283716605,
                             learning_rate=0.1,
                             # predictor='gpu_predictor'
                             )
    model.fit(trainX, trainy)
    # make a one-step prediction
    return model

def walk_forward_validation(data: np.array, n_test: int, target_type):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
  
        # fit and predict on the first iteration and every 20 iteration thereafter
        if (i % 5000 == 0) or (i == 0):
            model = xgboost_forecast(history, target_type)
            yhat = model.predict(np.asarray([testX]))
   
        # else just predict, to save time and memory
        else:
            yhat = model.predict(np.asarray([testX]))

        predictions.append(yhat)
        history.append(test[i])
        # summarize progress every 10 iterations
        if i % 1000 == 0:
            print(f"{i} > expected = {testy:.1f}, predicted = {yhat[0]:.1f}, error = {testy - yhat[0]:.1f}")
    
    # estimate prediction error
    if target_type == 'classifier':
    
    if target_type == 'regressor':
        error = mean_absolute_error(test[:, -1], predictions)
    return error, test[:, -1], predictions, model