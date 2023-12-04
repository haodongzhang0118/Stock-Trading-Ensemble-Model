import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.utils import seeding
import matplotlib.pyplot as plt
import matplotlib
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")

class StockEnvMine(gym.Env):

    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            df,
            hmax,
            initial_amount,
            num_stock_shares,
            buy_cost_pct,
            sell_cost_pct,
            state_space,
            stock_dim,
            tech_indicator_list,
            reward_scaling,
            action_space,
            initial=True,
            last_state=[],
            turbulence_th=None,
            plots=False,
            risk_indicator = 'turbulence',
            mode="",
            model_name="",
            iteration="",
    ):
        self.df = df
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.num_stock_shares = num_stock_shares
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.state_space = state_space
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(state_space,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_space,))
        self.stock_dim = stock_dim
        self.tech_indicator_list = tech_indicator_list
        self.reward_scaling = reward_scaling
        self.turbulence_th = turbulence_th
        self.plots = plots
        self.initial = initial
        self.terminal = False
        self.risk_indicator = risk_indicator
        self.state = self.initilize_state()
        self.log_every = 1
        self.mode = mode
        self.model_name = model_name
        self.iteration = iteration

        self.reward = 0
        self.turbulence= 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        self.asset_memory = [self.initial_amount + np.sum(np.array(self.num_stock_shares) * np.array(self.state[1:self.stock_dim+1]))]
        self.reward_memory = []
        self.actions_memory = []
        self.state_memory = ([])
        self.date_memory = [self.getDate()]
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def initilize_state(self):
        if self.initial:
            state = ([self.initial_amount] + self.data.close.values.tolist() + self.num_stock_shares + sum((self.data[tech].values.tolist() for tech in self.tech_indicator_list), []))
        else:
            state = ([self.last_state[0]] + self.data.close.values.tolist() + self.last_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)] + sum((self.data[tech].values.tolist() for tech in self.tech_indicator_list),[]))
        return state

    def getDate(self):
        return self.data.date.unique()[0]
    
    def render(self, mode="human", close=False):
        return self.state

    def reset(self, *, seed=None, options=None,):
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self.initilize_state()
        self.asset_memory = [self.initial_amount + np.sum(np.array(self.num_stock_shares) * np.array(self.state[1:self.stock_dim+1]))]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.reward_memory = []
        self.actions_memory = []
        self.date_memory = [self.getDate()]
        self.episode += 1
        return self.state, {}

    def update(self):
        state = ([self.state[0]] + self.data.close.values.tolist() + list(self.state[(self.stock_dim + 1) : (2 * self.stock_dim + 1)]) + sum((self.data[tech].values.tolist() for tech in self.tech_indicator_list), []))
        return state

    def buy(self, index, action):
        if self.state[index + 2 * self.stock_dim + 1] != True:
            nums_can_buy = self.state[0] // (self.state[index + 1] * (1 + self.buy_cost_pct[index]))
            nums = min(nums_can_buy, action)
            amount = self.state[index + 1] * (1 + self.buy_cost_pct[index]) * nums
            self.state[0] -= amount
            self.state[index + self.stock_dim + 1] += nums
            self.cost += amount
            self.trades += 1
        else:
            nums = 0
        return nums

    def Action_Buy(self, index, action):
        if self.turbulence_th is None:
            nums = self.buy(index, action)
        else:
            if self.turbulence < self.turbulence_th:
                nums = self.buy(index, action)
            else:
                nums = 0
        return nums

    def sell(self, index, action):
        if self.state[index + 2 * self.stock_dim + 1] != True:
            if self.state[index + self.stock_dim + 1] > 0:
                nums_can_sell = self.state[index + self.stock_dim + 1]
                nums = min(nums_can_sell, abs(action))
                amount = self.state[index + 1] * (1 - self.sell_cost_pct[index]) * nums
                self.state[0] += amount
                self.state[index + self.stock_dim + 1] -= nums
                self.cost += self.state[index + 1] * self.sell_cost_pct[index] * nums
                self.trades += 1
            else:
                nums = 0
        else:
            nums = 0
        return nums

    def Action_Sell(self, index, action):
        if self.turbulence_th is None:
            nums = self.sell(index, action)
        else:
            if self.turbulence < self.turbulence_th:
                nums = self.sell(index, action)
            else:
                if self.state[index + 1] > 0:
                    if self.state[index + self.stock_dim + 1] > 0:
                        nums = self.state[index + self.stock_dim + 1]
                        amount = self.state[index + 1] * (1 - self.sell_cost_pct[index]) * nums
                        self.state[0] += amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += self.state[index + 1] * self.sell_cost_pct[index] * nums
                        self.trades += 1
                    else:
                        nums = 0
                else:
                    nums = 0
        return nums

    def makePlot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def getDummyEnv(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def saveAssetMemory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame({"date": date_list, "account_value": asset_list})
        return df_account_value

    def saveActionMemory(self):
        date_list = self.date_memory[:-1]
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date
        return df_actions

    def step(self, actions):
        self.terminal = (self.day >= len(self.df.index.unique()) - 1 or self.state[0] <= 0)
        if self.terminal:
            if self.plots:
                self.makePlot()
            end_asset = self.state[0] + sum(np.array(self.state[1 : (self.stock_dim + 1)]) * np.array(self.state[(self.stock_dim + 1) : (2 * self.stock_dim + 1)]))
            df_total_value = pd.DataFrame(self.asset_memory)
            total_reward = self.state[0] + sum(np.array(self.state[1 : (self.stock_dim + 1)]) * np.array(self.state[(self.stock_dim + 1) : (2 * self.stock_dim + 1)])) - self.initial_amount
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
            if df_total_value["daily_return"].std() != 0:
                sharpe = (252 ** 0.5) * df_total_value["daily_return"].mean() / df_total_value["daily_return"].std()
            df_rewards = pd.DataFrame(self.reward_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.log_every == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_asset:0.2f}")
                print(f"total_reward: {total_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_total_value.to_csv("results/account_value_{}_{}_{}.csv".format(self.mode, self.model_name, self.iteration),index=False,)
                df_rewards.to_csv("results/account_rewards_{}_{}_{}.csv".format(self.mode, self.model_name, self.iteration), index=False,)
                plt.plot(self.asset_memory, "r")
                plt.savefig("results/account_value_{}_{}_{}.png".format(self.mode, self.model_name, self.iteration))
                plt.close()
            return self.state, self.reward, self.terminal, False, {}
        else:
            actions = (actions * self.hmax).astype(int)
            if self.turbulence_th is not None:
                if self.turbulence >= self.turbulence_th:
                    actions = np.array([-self.hmax] * self.stock_dim)
            begin_asset = self.state[0] + sum(np.array(self.state[1 : (self.stock_dim + 1)]) * np.array(self.state[(self.stock_dim + 1) : (2 * self.stock_dim + 1)]))
            sort_action = np.argsort(actions)
            sell_index = sort_action[:np.where(actions < 0)[0].shape[0]]
            buy_index = sort_action[::-1][:np.where(actions > 0)[0].shape[0]]
            for index in sell_index:
                actions[index] = self.Action_Sell(index, actions[index]) * (-1)
            for index in buy_index:
                actions[index] = self.Action_Buy(index, actions[index])
            self.actions_memory.append(actions)

            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.turbulence_th is not None:
                self.turbulence = self.data[self.risk_indicator].values[0]
            self.state = self.update()
            end_total_asset = self.state[0] + sum(np.array(self.state[1 : (self.stock_dim + 1)]) * np.array(self.state[(self.stock_dim + 1) : (2 * self.stock_dim + 1)]))
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self.getDate())
            # Reward Function
            self.reward = end_total_asset - begin_asset
            self.reward_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling
            self.state_memory.append(self.state)
            return self.state, self.reward, self.terminal, False, {}
