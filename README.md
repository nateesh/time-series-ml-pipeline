# Machine Learning pipeline for a time-series data

This repo is an exploration into how a tree-based ML algorithm can be implemented on typical financial time-series (TS) data (Open, High, Low, Close, Volume). The code was developed and tested with a CFD dataset of the S&P 500 (5min) time frame.

Technically any TS with OHLC datapoints could be used.

Key motivations:
- Get creative with feature engineering, beyond lag features
- Develop alternative features (VIX Futures, options)
- Exploring methods of training with limited GPU capacity 
- TODO: Develop alternative targets, beyond 'Close' *n* steps into the future
- TODO: Develop a specific way to evaluate and visualise what is considered successful