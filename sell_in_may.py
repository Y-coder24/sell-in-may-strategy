import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

vti = yf.download("VTI", start = "2005-01-01", auto_adjust=True)
retp = vti["Close"].pct_change().dropna().squeeze()
#vti: Close, High, Low, Open, Volume
print(vti.head())

posv = pd.Series(np.nan, index=retp.index)

datev = retp.index.strftime("%m-%d")
# Sell in May, Buy in November
posv[datev =="05-01"]=0
posv[datev =="05-03"]=0
posv[datev =="11-01"]=1
posv[datev =="11-03"]=1

#forward fill/backward fill
posv =posv.ffill()
posv = posv.bfill()

pnlmay = posv*retp
wealthv = pd.DataFrame({
    "VTI": retp,
    "sellmay": pnlmay
})

def sharpe_sortino(x):
  sharpe =np.sqrt(252)*x.mean()/x.std()
  downside_std =x[x<0].std()
  sortino = np.sqrt(252)*x.mean()/downside_std
  return pd.Series({"Sharpe": sharpe, "Sortino":sortino})

stats = wealthv.apply(sharpe_sortino)
print(stats)

# 累计收益
weekly_wealth = wealthv.cumsum().resample("W").last()
plt.figure(figsize=(10,6))
plt.plot(weekly_wealth["VTI"], label="VTI", color="blue")
plt.plot(weekly_wealth["sellmay"], label="Sell in May", color="red")
plt.title("Sell in May Strategy")
plt.legend()
plt.grid(True)
plt.show()