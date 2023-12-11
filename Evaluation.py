import datetime
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def RetrieveResults():
    for i in range(189, 505, 63):
        if i == 189:
            result_ppo = pd.read_csv(f"results/account_value_validation_ppo_ensemble_{i}.csv")
            result_a2c = pd.read_csv(f"results/account_value_validation_a2c_ensemble_{i}.csv")
            result_ddpg = pd.read_csv(f"results/account_value_validation_ddpg_ensemble_{i}.csv")
        else:
            result_ppo = pd.concat([result_ppo, pd.read_csv(f"results/account_value_validation_ppo_ensemble_{i}.csv")])
            result_a2c = pd.concat([result_a2c, pd.read_csv(f"results/account_value_validation_a2c_ensemble_{i}.csv")])
            result_ddpg = pd.concat(
                [result_ddpg, pd.read_csv(f"results/account_value_validation_ddpg_ensemble_{i}.csv")])
    result_ppo['adjusted_value'] = 1000000
    result_a2c['adjusted_value'] = 1000000
    result_ddpg['adjusted_value'] = 1000000
    result_ppo.reset_index(drop=True, inplace=True)
    result_a2c.reset_index(drop=True, inplace=True)
    result_ddpg.reset_index(drop=True, inplace=True)
    result_ppo['daily_return'].fillna(0, inplace=True)
    result_a2c['daily_return'].fillna(0, inplace=True)
    result_ddpg['daily_return'].fillna(0, inplace=True)
    for i in range(1, len(result_ppo)):
        result_ppo.loc[i, 'adjusted_value'] = result_ppo.loc[i - 1, 'adjusted_value'] * (
                1 + result_ppo.loc[i, 'daily_return'])
    for i in range(1, len(result_a2c)):
        result_a2c.loc[i, 'adjusted_value'] = result_a2c.loc[i - 1, 'adjusted_value'] * (
                1 + result_a2c.loc[i, 'daily_return'])
    for i in range(1, len(result_ddpg)):
        result_ddpg.loc[i, 'adjusted_value'] = result_ddpg.loc[i - 1, 'adjusted_value'] * (
                1 + result_ddpg.loc[i, 'daily_return'])
    result_ppo.set_index('date', inplace=True)
    result_a2c.set_index('date', inplace=True)
    result_ddpg.set_index('date', inplace=True)
    result_ddpg.index = pd.to_datetime(result_ddpg.index)
    result_ppo.index = pd.to_datetime(result_ppo.index)
    result_a2c.index = pd.to_datetime(result_a2c.index)
    return result_ppo, result_a2c, result_ddpg


def RetrieveBenchmark(start='2022-04-04', end='2023-09-01', ticker="^DJI"):
    # Download DJIA data from Yahoo Finance
    djia = yf.download(ticker, start=start, end=end)
    # Calculate daily returns
    djia['daily_return'] = djia['Adj Close'].pct_change()
    # Initialize the starting account balance
    initial_balance = 1_000_000
    # Calculate the cumulative return
    djia['cumulative_return'] = (1 + djia['daily_return']).cumprod()
    djia['adjusted_value'] = initial_balance * djia['cumulative_return']
    # Replace NaN values in 'adjusted_value' with the initial balance
    djia['adjusted_value'].fillna(initial_balance, inplace=True)
    djia.index = pd.to_datetime(djia.index)
    return djia


def ProcessEnsembleData(a):
    x_indices = a[a['account_value'] == 1000000].index[1:]
    for idx in x_indices:
        loc = a.index.get_loc(idx)
        prev_value = a.iloc[loc - 1]['account_value']
        next_value = a.iloc[loc + 1]['account_value']
        a.at[idx, 'account_value'] = (prev_value + next_value) / 2
    a = a.truncate(before=63)
    a['account_value'] /= (a['account_value'][0] / 1000000)
    a.set_index('date', inplace=True)
    a.index = pd.to_datetime(a.index)
    return a


def PlotResults(result_ppo, result_a2c, result_ddpg, djia, a):
    plt.figure(figsize=(20, 15))

    # Plot each DataFrame
    plt.plot(result_ddpg['adjusted_value'], label='DDPG')
    plt.plot(result_ppo['adjusted_value'], label='PPO')
    plt.plot(result_a2c['adjusted_value'], label='A2C')
    plt.plot(djia['adjusted_value'], label='DJIA(Benchmark)', linewidth=4.0)
    plt.plot(a['account_value'], label='Ensembled Agent', linewidth=4.0)

    # Adding titles and labels
    plt.title('Comparison of Account Balance')
    plt.xlabel('Date')
    plt.ylabel('Account Balance')

    # Add a legend
    plt.legend()
    plt.grid()
    # Show the plot
    plt.show()


def CalculateBeta(result_ppo, result_a2c, result_ddpg, djia, a):
    a['daily_return'] = a['account_value'].pct_change()
    covariance_a = a['daily_return'].cov(djia['daily_return'])
    variance_a = djia['daily_return'].var()
    beta_a = covariance_a / variance_a
    covariance_a2c = result_a2c['daily_return'].cov(djia['daily_return'])
    variance_a2c = djia['daily_return'].var()
    beta_a2c = covariance_a2c / variance_a2c
    covariance_ppo = result_ppo['daily_return'].cov(djia['daily_return'])
    variance_ppo = djia['daily_return'].var()
    beta_ppo = covariance_ppo / variance_ppo
    covariance_ddpg = result_ddpg['daily_return'].cov(djia['daily_return'])
    variance_ddpg = djia['daily_return'].var()
    beta_ddpg = covariance_ddpg / variance_ddpg
    beta_djia = 1
    return beta_a, beta_a2c, beta_ppo, beta_ddpg, beta_djia


def CalculateSharpe(result_ppo, result_a2c, result_ddpg, djia, a):
    rfr = 0.035
    daily_risk_free_return = (1 + rfr) ** (1 / 252) - 1  # assuming 252 trading days in a year
    sharpe_ratio_a = (a['daily_return'].mean() - daily_risk_free_return) / a['daily_return'].std() * (252 ** 0.5)
    sharpe_ratio_ppo = (result_ppo['daily_return'].mean() - daily_risk_free_return) / result_ppo[
        'daily_return'].std() * (252 ** 0.5)
    sharpe_ratio_a2c = (result_a2c['daily_return'].mean() - daily_risk_free_return) / result_a2c[
        'daily_return'].std() * (252 ** 0.5)
    sharpe_ratio_ddpg = (result_ddpg['daily_return'].mean() - daily_risk_free_return) / result_ddpg[
        'daily_return'].std() * (252 ** 0.5)
    sharpe_ratio_djia = (djia['daily_return'].mean() - daily_risk_free_return) / djia['daily_return'].std() * (
                252 ** 0.5)
    return sharpe_ratio_a, sharpe_ratio_a2c, sharpe_ratio_ppo, sharpe_ratio_ddpg, sharpe_ratio_djia


def CalculateCumulativeReturn(result_ppo, result_a2c, result_ddpg, djia, a):
    cumulative_return_a = a['account_value'].iloc[-1] / a['account_value'].iloc[0] - 1
    cumulative_return_ppo = result_ppo['account_value'].iloc[-1] / result_ppo['account_value'].iloc[0] - 1
    cumulative_return_a2c = result_a2c['account_value'].iloc[-1] / result_a2c['account_value'].iloc[0] - 1
    cumulative_return_ddpg = result_ddpg['account_value'].iloc[-1] / result_ddpg['account_value'].iloc[0] - 1
    cumulative_return_djia = djia['adjusted_value'].iloc[-1] / djia['adjusted_value'].iloc[0] - 1
    return cumulative_return_a, cumulative_return_a2c, cumulative_return_ppo, cumulative_return_ddpg, cumulative_return_djia


def CalculateMaxDrawdown(result_ppo, result_a2c, result_ddpg, djia, a):
    rolling_max_a = a['account_value'].cummax()
    daily_drawdown_a = a['account_value'] / rolling_max_a - 1.0
    max_drawdown_a = daily_drawdown_a.min()
    rolling_max_ppo = result_ppo['account_value'].cummax()
    daily_drawdown_ppo = result_ppo['account_value'] / rolling_max_ppo - 1.0
    max_drawdown_ppo = daily_drawdown_ppo.min()
    rolling_max_a2c = result_a2c['account_value'].cummax()
    daily_drawdown_a2c = result_a2c['account_value'] / rolling_max_a2c - 1.0
    max_drawdown_a2c = daily_drawdown_a2c.min()
    rolling_max_ddpg = result_ddpg['account_value'].cummax()
    daily_drawdown_ddpg = result_ddpg['account_value'] / rolling_max_ddpg - 1.0
    max_drawdown_ddpg = daily_drawdown_ddpg.min()
    rolling_max_djia = djia['adjusted_value'].cummax()
    daily_drawdown_djia = djia['adjusted_value'] / rolling_max_djia - 1.0
    max_drawdown_djia = daily_drawdown_djia.min()
    return max_drawdown_a, max_drawdown_a2c, max_drawdown_ppo, max_drawdown_ddpg, max_drawdown_djia


def PlotEvaluation(beta_values, sharpe_ratio_values, max_drawdown_values):
    names = ['Ensembled Agent', 'PPO', 'A2C', 'DDPG', 'DJIA(Benchmark)']

    # Function to add value labels
    def add_value_labels(ax, spacing=5):
        for rect in ax.patches:
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2

            label = "{:.2f}".format(y_value)
            ax.annotate(label, (x_value, y_value), xytext=(0, spacing),
                        textcoords="offset points", ha='center', va='bottom')

    # Plotting Beta Comparison
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()
    ax.bar(names, beta_values, color='skyblue')
    add_value_labels(ax)
    plt.title('Beta Comparison', fontsize=14)
    plt.ylabel('Beta', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

    # Plotting Sharpe Ratio Comparison
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()
    ax.bar(names, sharpe_ratio_values, color='lightgreen')
    add_value_labels(ax)
    plt.title('Sharpe Ratio Comparison', fontsize=14)
    plt.ylabel('Sharpe Ratio', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

    # Plotting Max Drawdown Comparison
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()
    ax.bar(names, max_drawdown_values, color='salmon')
    add_value_labels(ax)
    plt.title('Maximum Drawdown Comparison', fontsize=14)
    plt.ylabel('Maximum Drawdown', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()