import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 5]
import yfinance as yf
import mplfinance as mpf
import pendulum
from datetime import datetime as dt
from itertools import combinations
import statistics
import datetime
import mercury

import warnings
warnings.filterwarnings('ignore')

# Workflow:

# 1. Choose Pairs
# chosen_pairs = select_pairs(list of sp500 tickers, periods = ['1y'])

# 2. Backtest chosen pairs
# res = backtest_pairs(chosen_pairs = chosen_pairs, holding_period = 30, entry = 1.5, exit = 0.5, stop_loss = 2, periods = ['1y', '2y', '3y'])

#3. Track candidate pairs for trade signals
# tradable = track_pairs(list of candidate pairs, entry = 1.5, exit = 0.5, stop_loss = 2, display=False)

#4. Plot pairs
# plot_pairs(list of pairs)

def check_pairs(tickers, prices):
    pairs = list(combinations(tickers, 2))
    results = []
    for pair in range(len(pairs)):
        t1, t2 = pairs[pair][0], pairs[pair][1]
        prices_df = pd.merge(prices[t1], prices[t2], how = 'inner', left_index = True, right_index = True)
        prices_df = prices_df.dropna()
        corr = prices_df[t1].corr(prices_df[t2])
        coint_pval = coint(prices_df[t1], prices_df[t2])[1]
        adf_pval = adfuller(prices_df[t1]/prices_df[t2])[1]
        res = [round(corr,3), round(coint_pval,3), round(adf_pval,3)]
        results.append(res)
    df = pd.DataFrame(results, columns = ['Correlation', 'Cointegration', 'Stationarity'])
    df.index = pairs
    return df

def select_pairs(tickers, periods, corr_thresh = 0.75, coint_thresh = 0.05, stat_thresh = 0.05):
    chosen_pairs = []
    for period in periods:
        prices = yf.download(tickers = tickers, period = period, interval = '1d')['Close']
        df = check_pairs(tickers = tickers, prices = prices)
        df = df.sort_values(by = ['Correlation'], ascending = False)
        print(period)
        print("Initial Filtering... Cointegration p-value < 0.1 and Stationarity p-value < 0.1")
        print(df[(df["Cointegration"] < 0.1) & (df["Stationarity"] < 0.1)])
        chosen_pairs.append(df.index[(df["Correlation"] >= corr_thresh) &
                                     (df["Cointegration"] < coint_thresh) &
                                     (df["Stationarity"] < stat_thresh)].values)
    chosen_pairs = [pair for sublist in chosen_pairs for pair in sublist]
    return chosen_pairs

def backtest_pairs(chosen_pairs,
                   periods = ['1y', '2y', '3y'],
                   holding_period = 30,
                   entry = 1.5,
                   exit = 0.5,
                   stop_loss = 3):
    
    chosen_tickers = list(set([chosen_pairs for sublist in list(map(list, chosen_pairs)) for chosen_pairs in sublist]))
    
    res_df = pd.DataFrame()
    for period in periods:
        prices = yf.download(tickers = chosen_tickers, period = period, interval = '1d')['Close']
        for pair in chosen_pairs:
            stats = check_pairs(list(pair), prices)
            S1, S2 = [prices[pair[0]], prices[pair[1]]]
            ratio = S1/S2
            ratios_short = ratio.rolling(window=5, center=False).mean()
            ratios_long = ratio.rolling(window=20, center=False).mean()
            std_long = ratio.rolling(window=20, center=False).std()
            zscore = (ratios_short - ratios_long)/std_long

            portfolio = pd.DataFrame({S1.name: S1.values,
                      S2.name: S2.values,
                      'Ratio': ratio.values,
                      'Z_Score': zscore.values}, index = S1.index)

            strategy = []
            i = 0
            while i < len(portfolio):

                if abs(portfolio['Z_Score'][i]) > entry:
                    strategy.append('Enter')
                    i = i + 1

                    start, day = i, i
                    while day <= (start + holding_period):
                        if day == (start + holding_period):
                            strategy.append('Exit')
                            i = day + 1
                            break

                        elif day == len(portfolio):
                            break

                        elif abs(portfolio['Z_Score'][day]) > stop_loss:
                            strategy.append('Exit')
                            i = day + 1
                            break

                        elif abs(portfolio['Z_Score'][day]) < exit:
                            strategy.append('Exit')
                            i = day + 1
                            break

                        else:
                            strategy.append(0)
                            i = i + 1
                            day = day + 1

                else:
                    strategy.append(0)
                    i = i + 1

            portfolio['Strategy'] = strategy
            # portfolio.to_csv(str(period)+"_"+str(pair)+'.csv',index=True)
            trades = portfolio[(portfolio['Strategy'] == 'Enter') | (portfolio['Strategy'] == 'Exit')]
            res = []
            for i in range(len(trades)):

                if (trades['Strategy'][i] == 'Exit') and (trades['Z_Score'][i-1] < 1):
                    res.append(round(trades['Ratio'][i]/trades['Ratio'][i-1]-1, 4))

                elif (trades['Strategy'][i] == 'Exit') and (trades['Z_Score'][i-1] > 1):
                    res.append(-1*round(trades['Ratio'][i]/trades['Ratio'][i-1]-1,4))

                else:
                    res.append(np.nan)

            trades["Returns"] = res      

            res = [x for x in res if ~np.isnan(x)] 
            if len(res) != 0:
                total_return = round(sum(res),4)
                avg_return = round(np.mean(res),4)
                n = len(res)
                accuracy = round(sum([i > 0 for i in res])/len(res),4)
            else:
                total_return, avg_return, n, accuracy = [np.nan, np.nan, np.nan, np.nan]
            
            if sum(trades['Strategy'] == "Exit") == sum(trades['Strategy'] == "Enter"):
                count = list((trades.index[trades.Strategy == "Exit"] - trades.index[trades.Strategy == "Enter"]).days)
            else:
                count = list((trades.index[trades.Strategy == "Exit"] - trades.index[trades.Strategy == "Enter"][:-1]).days)

            res_df = res_df.append({"year": period,
                                    "pair": pair,
                                    "total_return": total_return,
                                    "avg_return": avg_return,
                                    "trades": n,
                                    "accuracy": accuracy,
                                    "avg_duration": round(np.mean(count),2),
                                    "correlation": stats['Correlation'].values[0],
                                    "cointegration": stats['Cointegration'].values[0],
                                    "stationarity": stats['Stationarity'].values[0]}, ignore_index = True)
    
    with pd.ExcelWriter("backtest_results.xlsx") as writer:
        if isinstance(res_df, pd.DataFrame):
            res = res_df.sort_values(by = ['correlation'], ascending = False)
            filtered = res[(res['cointegration'] < 0.05) & (res["stationarity"] < 0.05)]
            filtered.to_excel(writer, sheet_name = "Filtered", index = False)
            res.to_excel(writer, sheet_name = "All", index = False)
        else:
            print('No viable pairs!')

        return res_df if len(res_df) != 0 else 'No viable pairs!'
def track_pairs(pairs, entry = 1.5, exit = 0.5, stop_loss = 2, display = False):
    tradable = []
    for pair in pairs:
        T1, T2 = pair[0], pair[1]
        prices = yf.download(tickers = [T1, T2], period = '1y', interval = '1d')['Close']
        ratio = prices[T1]/prices[T2]
        ratios_short = ratio.rolling(window=5, center=False).mean()
        ratios_long = ratio.rolling(window=20, center=False).mean()
        std = ratio.rolling(window=20, center=False).std()
        zscore = (ratios_short - ratios_long)/std
        if (zscore.tail(1).values[0] > abs(entry)) or display == True:     
            zscore.plot()
            plt.title(T1 + " to " + T2 + " Z-Score", size = 14)
            plt.axhline(0, color='black')
            #plt.axhline(1, color='blue', linestyle='--')
            #plt.axhline(-1, color='blue', linestyle='--')
            plt.axhline(entry, color='green', linestyle='--')
            plt.axhline(-entry, color='green', linestyle='--')
            plt.axhline(exit, color='red', linestyle='--')
            plt.axhline(-exit, color='red', linestyle='--')
            plt.ylim(-stop_loss, stop_loss)
            plt.annotate('Z-Score on ' + str(zscore.tail(1).index.month[0]) + "/" + str(zscore.tail(1).index.day[0]) +
                         ": " + str(round(zscore.tail(1).values[0],2)),
                        xy=(0.5, 0), xytext=(0, 2), size = 12,
                        xycoords=('axes fraction', 'figure fraction'),
                        textcoords='offset points', ha='center', va='bottom')
            plt.show()
            tradable.append(pair)
        else:
            print("No Trade Signal!")
    return tradable

def plot_pairs(pairs):
    for pair in pairs:
        T1, T2 = pair[0], pair[1]
        prices = yf.download(tickers = [T1, T2], period = '1y', interval = '1d')['Close']
        fig, ax1 = plt.subplots(figsize=(15, 5))
        plt.title(T1 + " vs. " + T2)
        ax2 = ax1.twinx()
        ax1.plot(prices[T1], "blue")
        ax2.plot(prices[T2], "red")
        ax1.legend([T1], loc = "upper left")
        ax2.legend([T2], loc = "upper right")
        plt.show()


