📈 Quantitative Trading Strategy using OLS, Ridge, and Lasso

Overview

We compare:

Ordinary Least Squares (OLS)
Ridge Regression
Lasso Regression

The goal is to evaluate predictive performance and assess whether linear models can generate profitable trading signals.

Data

Source: Yahoo Finance (via yfinance)
Asset: AAPL
Period: 2010-2025

Features Engineered

Lagged returns (1, 2, 3 days)
Moving averages (5, 10 days)
RSI (Relative Strength Index)
MACD (Moving Average Convergence Divergence)
Rolling volatility (10-day)
Volume-based features

Target

Next-day log return

Models Used

Linear Regression (OLS)
Ridge Regression (L2 regularization)
Lasso Regression (L1 regularization)

Evaluation Metrics

RMSE
R² score
Directional Accuracy
Sharpe Ratio
Statistical significance

Strategy
Go long if predicted return > 0
Go short if predicted return < 0

Results

Compared model performance across OLS, Ridge, and Lasso
Lasso used for implicit feature selection
Evaluated cumulative returns and Sharpe ratio

Key Insights

Regularization helps reduce noise in financial data
Lasso identifies most relevant predictors
Linear models have limited predictive power but can provide directional signals

Future Improvements
Add transaction costs
Use multiple assets (pairs trading / cross-sectional)

Tech Stack involved
Python
Pandas, NumPy
Scikit-learn
Matplotlib
SciPy
