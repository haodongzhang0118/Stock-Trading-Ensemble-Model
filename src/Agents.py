import numpy as np
import pandas as pd
from finrl import config
from finrl.agents.stablebaselines3.models import TensorboardCallback
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3


class Agent:
    def __init__(self, env, model_name, policy="MlpPolicy", policy_kwargs=None, model_kwargs=None, verbose=1, seed=None, tensorboard_log=None):
        self.models = {"a2c": A2C, "ddpg": DDPG, "ppo": PPO, "td3": TD3, "sac": SAC}
        model_kwargs_dict = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in self.models.keys()}
        self.model_name = model_name
        if model_kwargs is None:
            self.model_kwargs = model_kwargs_dict[model_name]
        else:
            self.model_kwargs = model_kwargs
        self.model = self.models[model_name](policy=policy, env=env, verbose=verbose, seed=seed, tensorboard_log=tensorboard_log, policy_kwargs=policy_kwargs, **self.model_kwargs)

    def train(self, total_timesteps=5000):
        model = self.model.learn(total_timesteps=total_timesteps, tb_log_name=self.model_name, callback=TensorboardCallback())
        return model

    def predict(self, env_new, deterministic=True):
        env, obs = env_new.getDummyEnv()
        account_memory = None
        actions_memory = None

        env.reset()
        max_step = len(env_new.df.index.unique()) - 1

        for i in range(max_step + 1):
            action, states = self.model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, info = env.step(action)
            if i == max_step - 1:
                account_memory = env.env_method(method_name="saveAssetMemory")
                actions_memory = env.env_method(method_name="saveActionMemory")
            if dones[0]:
                print("Finished")
                break
        return account_memory[0], actions_memory[0]

    def predictLoadFromFile(self, env_new, cwd, deterministic=True):
        try:
            model = self.model.load(cwd)
            print("Model loaded from file")
        except BaseException as error:
            raise ValueError(f"Failed to load agent. Error: {str(error)}") from error

        state = env_new.reset()
        episode_returns = []
        episode_total_assets = [env_new.initial_amount]
        done = False
        while not done:
            action = model.predict(state, deterministic=deterministic)[0]
            state, reward, done, _ = env_new.step(action)
            episode_total_assets.append(state[0])
            episode_return = state[0] / env_new.initial_amount
            episode_returns.append(episode_return)
        print(f"Finish Trading. The final amount of money is: {episode_total_assets[-1]}. The total return is: {episode_returns[-1]}")
        return episode_total_assets, episode_returns