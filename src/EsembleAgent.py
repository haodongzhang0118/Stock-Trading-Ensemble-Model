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
    def __init__(self, df, train_period, val_period, rebalance_window, validation_window, env_args):
        self.df = df
        self.train_period = train_period
        self.val_period = val_period
        self.rebalance_window = rebalance_window
        self.validation_window = validation_window
        self.env_args = env_args
        self.unique_trade_date = df[(df.date > val_period[0]) & (df.date <= val_period[1])].date.unique()
        self.train_env = None

    def val(self, model, val_data, val_env, val_obs):
        for _ in range(len(val_data.index.unique())):
            action, _states = model.predict(val_obs)
            val_obs, rewards, dones, info = val_env.step(action)

    def predict(self, model, name, last_state, iter, tur_th, initial):
        trade_data = data_split(self.df, start=self.unique_trade_date[iter - self.rebalance_window], end=self.unique_trade_date[iter],)
        trade_env = DummyVecEnv([lambda: StockEnvMine(df=trade_data, turbulence_th=tur_th, iteration=iter, mode="trade", model_name=name, last_state=last_state, initial=initial, **self.env_args)])
        trade_obs = trade_env.reset()
        for i in range(len(trade_data.index.unique())):
            action, _ = model.predict(trade_obs)
            trade_obs, _, _, _ = trade_env.step(action)
            if i == (len(trade_data.index.unique()) - 2):
                last_state = trade_env.envs[0].render()

        df_last_state = pd.DataFrame({"last_state": last_state})
        df_last_state.to_csv(f"results/last_state_{name}_{i}.csv", index=False)
        return last_state

    def getSharpe(self, iter, model_name):
        df_total_value = pd.read_csv(f"results/account_value_validation_{model_name}_{iter}.csv")
        if df_total_value["daily_return"].var() == 0:
            if df_total_value["daily_return"].mean() > 0:
                return np.inf
            else:
                return 0
        else:
            return df_total_value["daily_return"].mean() / df_total_value["daily_return"].std() * np.sqrt(4)

    def train(self, A2C_kwargs=None, PPO_kwargs=None, DDPG_kwargs=None, timesteps={"a2c": 50000, "ppo": 50000, "ddpg": 50000}):
        tell = True
        a2c_sharpe = []
        ddpg_sharpe = []
        ppo_sharpe = []
        last_state = []

        model_order = []
        val_start_date = []
        val_end_date = []
        iteration_list = []

        insample_turbulence = self.df[(self.df.date >= self.train_period[0]) & (self.df.date < self.train_period[1])]
        insample_tur_threshold = np.quantile(insample_turbulence.turbulence.values, .90)
        for i in range(self.rebalance_window + self.validation_window, len(self.unique_trade_date) + self.rebalance_window + self.validation_window, self.rebalance_window):
            val_start = self.unique_trade_date[i - self.rebalance_window - self.validation_window]
            if i > len(self.unique_trade_date):
              tell = False
            if i - self.rebalance_window > len(self.unique_trade_date):
              end_index = -1
            else:
              end_index = i - self.rebalance_window
            val_end = self.unique_trade_date[end_index]
            # val_start = self.unique_trade_date[i]
            # val_end = self.unique_trade_date[i + self.rebalance_window]
            val_start_date.append(val_start)
            val_end_date.append(val_end)
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

            print("Ensemble Model Training: ")
            if (sharpe_a2c > sharpe_ppo) & (sharpe_a2c > sharpe_ddpg):
                model_order.append("a2c")
                model_ensemble = agent_a2c.model
            elif (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_ddpg):
                model_order.append("ppo")
                model_ensemble = agent_ppo.model
            else:
                model_order.append("ddpg")
                model_ensemble = agent_ddpg.model

            if tell:
                last_state = self.predict(model=model_ensemble, name="ensemble", last_state=last_state, iter=i, tur_th=tur_threshold, initial=initial)

        df_summary = pd.DataFrame({"iteration": iteration_list, "Start Date": val_start_date, "End Date": val_end_date, "model_order": model_order, "a2c_sharpe": a2c_sharpe, "ddpg_sharpe": ddpg_sharpe, "ppo_sharpe": ppo_sharpe})
        return df_summary
