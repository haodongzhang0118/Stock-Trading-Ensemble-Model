import numpy as np
import pandas as pd
from finrl import config
from finrl.agents.stablebaselines3.models import TensorboardCallback
from finrl.meta.preprocessor.preprocessors import data_split
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stockEnv import StockEnvMine
from Agents import Agent

class EnsembleAgent:
    def __init__(self, df, train_period, val_period, rebalance_window, validation_window, env_args, market_ticker="^GSPC"):
        self.df = df
        self.train_period = train_period
        self.val_period = val_period
        self.rebalance_window = rebalance_window
        self.validation_window = validation_window
        self.env_args = env_args
        self.unique_trade_date = df[(df.date > val_period[0]) & (df.date <= val_period[1])].date.unique()
        self.train_env = None
        self.val_start_date = []
        self.val_end_date = []
        self.tur_nums = []
        self.agents_order = []

        columns = ['Iteration', 'Yield', 'Sharpe', 'Beta']
        self.a2c_metrics = pd.DataFrame(columns=columns)
        self.ddpg_metrics = pd.DataFrame(columns=columns)
        self.ppo_metrics = pd.DataFrame(columns=columns)
        self.ensembled_metrics = pd.DataFrame(columns=columns)
        self.market_ticker = market_ticker
        self.market_returns = self.download_market_returns()

    def download_market_returns(self):
        start_date = self.train_period[0]
        end_date = self.val_period[1]

        # Downloading market data
        market_data = yf.download(self.market_ticker, start=start_date, end=end_date)

        # Calculating daily returns
        market_return_df = market_data.reset_index()
        market_return_df.rename(columns={'Date': 'date'}, inplace=True)
        market_return_df['date'] = pd.to_datetime(market_return_df['date'], format='%Y-%m-%d')
        market_return_df['market_return'] = market_return_df['Adj Close'].pct_change(1)
        return market_return_df


    def val(self, model, val_data, val_env, val_obs):
        for _ in range(len(val_data.index.unique())):
            action, _states = model.predict(val_obs)
            val_obs, rewards, dones, info = val_env.step(action)

    def predict(self):
        last_state = None
        initial = True
        deterministic = True
        account_list = []
        actions_list = []
        for i in range(len(self.agents_order)):
            trade_data = data_split(self.df, start=self.val_start_date[i], end=self.val_end_date[i],)
            trade_env = StockEnvMine(df=trade_data, turbulence_th=self.tur_nums[i], iteration=iter, mode="trade", model_name=self.agents_order[i].model_name, last_state=last_state, initial=initial, **self.env_args)
            env, obs = trade_env.getDummyEnv()
            account_memory = None
            actions_memory = None
            env.reset()
            max_step = len(trade_env.df.index.unique()) - 1
            for j in range(max_step + 1):
                action, states = self.agents_order[i].model.predict(obs, deterministic=deterministic)
                obs, rewards, dones, info = env.step(action)
                if j == max_step - 1:
                    account_memory = env.env_method(method_name="saveAssetMemory")
                    actions_memory = env.env_method(method_name="saveActionMemory")
                    last_state = env.envs[0].render()
                if dones[0]:
                    print("Finished")
                    break
            if i == 0:
              initial = False
            account_list.append(account_memory[0])
            actions_list.append(actions_memory[0])
        return pd.concat(account_list, ignore_index=True), pd.concat(actions_list, ignore_index=True)

    def getSharpe(self, iter, model_name):
      df_total_value = pd.read_csv(f"results/account_value_validation_{model_name}_{iter}.csv")
      if df_total_value["daily_return"].var() == 0:
          if df_total_value["daily_return"].mean() > 0:
              return np.inf
          else:
              return 0
      else:
          return df_total_value["daily_return"].mean() / df_total_value["daily_return"].std() * np.sqrt(4)

    def getPeriodReturn(self, iter, model_name):
      if model_name == "trade_ensemble":
        df_total_value = pd.read_csv(f"results/account_value_{model_name}_{iter}.csv")
      else:
        df_total_value = pd.read_csv(f"results/account_value_validation_{model_name}_{iter}.csv")
      start_value = df_total_value["account_value"].iloc[0]
      end_value = df_total_value["account_value"].iloc[-1]
      period_return = (end_value - start_value) / start_value
      return period_return

    def getPeriodBeta(self, iter, model_name):
      if model_name == "trade_ensemble":
        df = pd.read_csv(f"results/account_value_{model_name}_{iter}.csv")
      else:
        df = pd.read_csv(f"results/account_value_validation_{model_name}_{iter}.csv")
      df['daily_return'] = df['account_value'].pct_change(1)

      # Ensure alignment with rebalance period
      df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
      merged = df.merge(self.market_returns, on='date', how='inner')

      # Calculate covariance and variance for the period
      covariance = merged['daily_return'].cov(merged['market_return'])
      market_variance = merged['market_return'].var()

      beta = covariance / market_variance
      return beta

    def getPeriodSharpe(self, iter, model_name, risk_free_rate=0.035/256):
      if model_name == "trade_ensemble":
        df_total_value = pd.read_csv(f"results/account_value_{model_name}_{iter}.csv")
      else:
        df_total_value = pd.read_csv(f"results/account_value_validation_{model_name}_{iter}.csv")
      df_total_value['daily_return'] = df_total_value['account_value'].pct_change(1)

      # Calculate excess returns over the risk-free rate
      excess_returns = df_total_value['daily_return'] - risk_free_rate

      # Calculate average excess return and standard deviation of returns
      avg_excess_return = excess_returns.mean() * 63
      std_dev = excess_returns.std() * np.sqrt(63)

      # Handle the case where standard deviation is zero
      if std_dev == 0:
          return np.inf if avg_excess_return > 0 else 0

      # Sharpe Ratio for the period (not annualized)
      sharpe_ratio = avg_excess_return / std_dev
      return sharpe_ratio



    def train(self, A2C_kwargs=None, PPO_kwargs=None, DDPG_kwargs=None, timesteps={"a2c": 50000, "ppo": 50000, "ddpg": 50000}):
        tell = True
        a2c_sharpe = []
        ddpg_sharpe = []
        ppo_sharpe = []
        last_state = []

        model_order = []
        iteration_list = []

        insample_turbulence = self.df[(self.df.date >= self.train_period[0]) & (self.df.date < self.train_period[1])]
        insample_tur_threshold = np.quantile(insample_turbulence.turbulence.values, .90)
        for i in range(self.rebalance_window + self.validation_window, len(self.unique_trade_date) + self.rebalance_window + self.validation_window, self.rebalance_window):
            val_start = self.unique_trade_date[i - self.rebalance_window - self.validation_window]
            if i - self.rebalance_window > len(self.unique_trade_date):
              end_index = -1
            else:
              end_index = i - self.rebalance_window
            val_end = self.unique_trade_date[end_index]
            self.val_start_date.append(val_start)
            self.val_end_date.append(val_end)
            iteration_list.append(i)
            initial = (i - self.rebalance_window - self.validation_window == 0)
            end_date = self.df.index[self.df["date"] == self.unique_trade_date[i - self.rebalance_window - self.validation_window]].to_list()[-1]
            start_date = end_date - 63 + 1
            history_tur_mean = np.mean(self.df.iloc[start_date : (end_date + 1), :].drop_duplicates(subset=["date"]).turbulence.values)
            if history_tur_mean > insample_tur_threshold:
                tur_threshold = insample_tur_threshold
            else:
                tur_threshold = np.quantile(insample_turbulence.turbulence.values, 0.99)
            print("Turbulence threshold: ", tur_threshold)
            self.tur_nums.append(tur_threshold)

            train = data_split(self.df, start=self.train_period[0], end=self.unique_trade_date[i - self.rebalance_window - self.validation_window],)
            validation = data_split(self.df, start=self.unique_trade_date[i - self.rebalance_window - self.validation_window], end=self.unique_trade_date[end_index],)
            self.train_env = DummyVecEnv([lambda: StockEnvMine(df=train, **self.env_args)])
            print("Model training from: {} to {}".format(self.train_period[0], self.unique_trade_date[i - self.rebalance_window - self.validation_window]))
            print("A2C Training: ")
            agent_a2c = Agent(env=self.train_env, iter_num=i, model_name="a2c_ensemble", model_kwargs=A2C_kwargs)
            trained_a2c = agent_a2c.train(total_timesteps=timesteps["a2c"])
            agent_a2c.model = trained_a2c
            print("A2C Validation from {} to {}".format(val_start, val_end))
            val_env_a2c = DummyVecEnv([lambda: StockEnvMine(df=validation, turbulence_th=tur_threshold, iteration=i, mode="validation", model_name="a2c_ensemble", **self.env_args)])
            val_obs_a2c = val_env_a2c.reset()
            self.val(agent_a2c.model, validation, val_env_a2c, val_obs_a2c)
            sharpe_a2c = self.getSharpe(i, model_name="a2c_ensemble")
            a2c_sharpe.append(sharpe_a2c)
            period_return_a2c = self.getPeriodReturn(i, "a2c_ensemble")
            period_sharpe_a2c = self.getPeriodSharpe(i, "a2c_ensemble")
            period_beta_a2c = self.getPeriodBeta(i, "a2c_ensemble")
            new_row_a2c = pd.DataFrame({'Iteration': [i], 'Yield': [period_return_a2c], 'Sharpe': [period_sharpe_a2c], 'Beta': [period_beta_a2c]})
            self.a2c_metrics = pd.concat([self.a2c_metrics, new_row_a2c], ignore_index=True)


            print("DDPG Training: ")
            agent_ddpg = Agent(env=self.train_env, iter_num=i, model_name="ddpg_ensemble", model_kwargs=DDPG_kwargs)
            trained_ddpg = agent_ddpg.train(total_timesteps=timesteps["ddpg"])
            agent_ddpg.model = trained_ddpg
            print("DDPG Validation from {} to {}".format(val_start, val_end))
            val_env_ddpg = DummyVecEnv([lambda: StockEnvMine(df=validation, turbulence_th=tur_threshold, iteration=i, mode="validation", model_name="ddpg_ensemble", **self.env_args)])
            val_obs_ddpg = val_env_ddpg.reset()
            self.val(agent_ddpg.model, validation, val_env_ddpg, val_obs_ddpg)
            sharpe_ddpg = self.getSharpe(i, model_name="ddpg_ensemble")
            ddpg_sharpe.append(sharpe_ddpg)
            period_return_ddpg = self.getPeriodReturn(i, "ddpg_ensemble")
            period_sharpe_ddpg = self.getPeriodSharpe(i, "ddpg_ensemble")
            period_beta_ddpg = self.getPeriodBeta(i, "ddpg_ensemble")
            new_row_ddpg = pd.DataFrame({'Iteration': [i], 'Yield': [period_return_ddpg], 'Sharpe': [period_sharpe_ddpg], 'Beta': [period_beta_ddpg]})
            self.ddpg_metrics = pd.concat([self.ddpg_metrics, new_row_ddpg], ignore_index=True)


            print("PPO Training: ")
            agent_ppo = Agent(env=self.train_env, iter_num=i, model_name="ppo_ensemble", model_kwargs=PPO_kwargs)
            trained_ppo = agent_ppo.train(total_timesteps=timesteps["ppo"])
            agent_ppo.model = trained_ppo
            print("PPO Validation from {} to {}".format(val_start, val_end))
            val_env_ppo = DummyVecEnv([lambda: StockEnvMine(df=validation, turbulence_th=tur_threshold, iteration=i, mode="validation", model_name="ppo_ensemble", **self.env_args)])
            val_obs_ppo = val_env_ppo.reset()
            self.val(agent_ppo.model, validation, val_env_ppo, val_obs_ppo)
            sharpe_ppo = self.getSharpe(i, model_name="ppo_ensemble")
            ppo_sharpe.append(sharpe_ppo)
            period_return_ppo = self.getPeriodReturn(i, "ppo_ensemble")
            period_sharpe_ppo = self.getPeriodSharpe(i, "ppo_ensemble")
            period_beta_ppo = self.getPeriodBeta(i, "ppo_ensemble")
            new_row_ppo = pd.DataFrame({'Iteration': [i], 'Yield': [period_return_ppo], 'Sharpe': [period_sharpe_ppo], 'Beta': [period_beta_ppo]})
            self.ppo_metrics = pd.concat([self.ppo_metrics, new_row_ppo], ignore_index=True)

            choosen_model = None
            print("Ensemble Model Training: ")
            if (sharpe_a2c > sharpe_ppo) & (sharpe_a2c > sharpe_ddpg):
                model_order.append("a2c")
                self.agents_order.append(agent_a2c)
                choosen_model = "a2c_ensemble"
            elif (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_ddpg):
                model_order.append("ppo")
                self.agents_order.append(agent_ppo)
                choosen_model = "ppo_ensemble"
            else:
                model_order.append("ddpg")
                self.agents_order.append(agent_ddpg)
                choosen_model = "ddpg_ensemble"


            period_return_ensemble = self.getPeriodReturn(i, choosen_model)
            period_sharpe_ensemble = self.getPeriodSharpe(i, choosen_model)
            period_beta_ensemble = self.getPeriodBeta(i, choosen_model)
            new_row_ensemble = pd.DataFrame({'Iteration': [i], 'Yield': [period_return_ensemble], 'Sharpe': [period_sharpe_ensemble], 'Beta': [period_beta_ensemble]})
            self.ensembled_metrics = pd.concat([self.ensembled_metrics, new_row_ensemble], ignore_index=True)

        df_summary = pd.DataFrame({"iteration": iteration_list, "Start Date": self.val_start_date, "End Date": self.val_end_date, "model_order": model_order, "a2c_sharpe": a2c_sharpe, "ddpg_sharpe": ddpg_sharpe, "ppo_sharpe": ppo_sharpe})
        return df_summary, self.a2c_metrics, self.ddpg_metrics, self.ppo_metrics, self.ensembled_metrics
