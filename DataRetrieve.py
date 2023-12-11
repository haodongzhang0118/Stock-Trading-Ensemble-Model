import condacolab
condacolab.install()
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import talib
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv


def RetrieveTrainingData(start_date = "2021-01-01", end_date = "2023-09-01"):
    # Initialize a list to store the data
    all_data = []

    for ticker in DOW_30_TICKER:
        # Download data
        if ticker == "DOW":
            continue
        data = yf.download(ticker, start=start_date, end=end_date)

        # Calculate Technical Indicators
        data['MACD'], data['MACDSignal'], data['MACDHist'] = talib.MACD(data['Close'], fastperiod=2, slowperiod=3,
                                                                        signalperiod=4)
        data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
        data['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)
        data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
        data.rename(columns={'Close': 'close'}, inplace=True)

        # Add ticker and reset index
        data['tic'] = ticker
        data.reset_index(inplace=True)

        # Append to the list
        all_data.append(data)

    # Combine into a single DataFrame
    combined_data = pd.concat(all_data)

    # Sort by date and reset index
    combined_data.sort_values(by='Date', inplace=True)
    combined_data.rename(columns={'Date': 'date'}, inplace=True)
    combined_data.reset_index(drop=True, inplace=True)

    # Select relevant columns
    final_df = combined_data[['date', 'tic', 'close', 'Adj Close', 'MACD', 'RSI', 'CCI', 'ADX']]

    final_df.reset_index()
    final_df['index'] = final_df.groupby('date').ngroup()
    final_df.set_index('index', inplace=True)
    reward_scaling = 1e-4
    Indicators = ['Adj Close', 'MACD', 'RSI', 'CCI', 'ADX']
    stock_dim = len(final_df.tic.unique())
    state_space = 1 + 2 * stock_dim + len(Indicators) * stock_dim
    InitialAmount = 1000000
    num_stock_holders = [0] * stock_dim
    buy_cost = [0.001] * stock_dim
    sell_cost = [0.001] * stock_dim
    print("""There are {} stocks in the dataset.
    The state space we used for training is {}.
    The indicators we used is {}.
    The reward scaling is {}.
    The initial amount is {}.
    The share for each storck in the beginning is 0.
    Cost of buying and selling is 0.001.
    """.format(stock_dim, state_space, Indicators, reward_scaling, InitialAmount))
    return final_df
