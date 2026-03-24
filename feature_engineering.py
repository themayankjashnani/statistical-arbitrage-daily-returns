from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data/stock_data.csv')
df["Date"] = pd.to_datetime(df["Date"], format="ISO8601", utc=True)
df.set_index("Date", inplace=True)
df.index = df.index.tz_convert(None)

def rsi(close=df['Close'], period=14):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def macd(close=df['Close'], fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line

    return macd_line, signal_line, hist

df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
df["ret_lag1"] = df["log_return"].shift(1)
df["ret_lag2"] = df["log_return"].shift(2)
df["ret_lag3"] = df["log_return"].shift(3)

df["ma_5"] = df["log_return"].rolling(window=5).mean()
df["ma_10"] = df["log_return"].rolling(window=10).mean()
df["ma_ratio"] = df["ma_5"] / df["ma_10"]

df["rsi"] = rsi(df["Close"], period=14)
df["macd"] = macd(df["Close"])[0]

df["vol_10"] = df["log_return"].rolling(window=10).std()
df["vol_chg"] = df["Volume"].pct_change()
df["vol_ma"] = df["Volume"].rolling(window=10).mean()

df["target"] = df["log_return"].shift(-1)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# we are predicting next day's return based on previous day's return (i+1 using i,i-1,i-2)
df_train = df.loc[:'2018']
df_test  = df.loc['2019':]
feature_cols = [
    "ret_lag1", "ret_lag2", "ret_lag3",
    "ma_5", "ma_10", "ma_ratio",
    "rsi", "macd",
    "vol_10", "vol_chg", "vol_ma"
]

X_train = df_train[feature_cols]
y_train = df_train["target"]

X_test = df_test[feature_cols]
y_test = df_test["target"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(X_train.shape)
print(X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(y_pred[:5])

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R²:", r2)

directional_acc = np.mean(
    np.sign(y_pred) == np.sign(y_test)
)

print("Directional Accuracy:", directional_acc)

strategy_returns = np.sign(y_pred) * y_test
cum_returns = np.cumsum(strategy_returns)

plt.plot(cum_returns)
plt.title("Cumulative Strategy Returns")
plt.show()

from scipy.stats import binomtest

p_value = binomtest(
    np.sum(np.sign(y_pred) == np.sign(y_test)),
    n=len(y_test),
    p=0.5,
    alternative='greater'
)

print("p-value:", p_value)

from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

lasso = Lasso(alpha=0.001)
lasso.fit(X_train_scaled, y_train)

y_pred_ridge = ridge.predict(X_test_scaled)
y_pred_lasso = lasso.predict(X_test_scaled)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
print("Ridge RMSE:", rmse_ridge)
print("Lasso RMSE:", rmse_lasso)
directional_acc_ridge = np.mean(
    np.sign(y_pred_ridge) == np.sign(y_test)
)
directional_acc_lasso = np.mean(
    np.sign(y_pred_lasso) == np.sign(y_test)
)
print("Ridge Directional Accuracy:", directional_acc_ridge)
print("Lasso Directional Accuracy:", directional_acc_lasso) 

strategy_returns = np.sign(y_pred) * y_test
cum_returns = strategy_returns.cumsum()

plt.plot(cum_returns)
plt.title("Cumulative Strategy Returns")
plt.show()

sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
print("Sharpe Ratio:", sharpe)

from scipy.stats import binomtest
import numpy as np

successes = np.sum(np.sign(y_pred_lasso) == np.sign(y_test))
n = len(y_test)

result = binomtest(successes, n, p=0.5, alternative="greater")
print(result.pvalue)
lasso_coefs = pd.Series(
    lasso.coef_,
    index=X_train.columns
).sort_values(key=abs, ascending=False)

print(lasso_coefs)

strategy_returns = np.sign(y_pred_lasso) * y_test
cum_returns = strategy_returns.cumsum()

plt.plot(cum_returns)
plt.title("Lasso Strategy Cumulative Returns")
plt.show()

