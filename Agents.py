import numpy as np
import pandas as pd
from finrl import config
from finrl.agents.stablebaselines3.models import TensorboardCallback
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO


class Agent:
    def __init__(self, env, model_name, iter_num=0, policy="MlpPolicy", policy_kwargs=None, model_kwargs=None, verbose=1, seed=None, tensorboard_log=None):
        self.models = {"a2c": A2C, "ddpg": DDPG, "ppo": PPO, "a2c_ensemble": A2C, "ddpg_ensemble": DDPG, "ppo_ensemble": PPO}
        model_kwargs_dict = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in ["a2c", "ddpg", "ppo"]}
        model_kwargs_dict_ensemble = {x + "_ensemble": config.__dict__[f"{x.upper()}_PARAMS"] for x in ["a2c", "ddpg", "ppo"]}
        model_kwargs_dict.update(model_kwargs_dict_ensemble)
        self.model_name = model_name
        self.iter_num = iter_num
        if model_kwargs is None:
            self.model_kwargs = model_kwargs_dict[model_name]
        else:
            self.model_kwargs = model_kwargs
        self.model = self.models[model_name](policy=policy, env=env, verbose=verbose, seed=seed, tensorboard_log=tensorboard_log, policy_kwargs=policy_kwargs, **self.model_kwargs)

    def train(self, total_timesteps=5000):
        model = self.model.learn(total_timesteps=total_timesteps, tb_log_name="{}_{}".format(self.model_name, self.iter_num), callback=TensorboardCallback())
        model.save(f"{config.TRAINED_MODEL_DIR}/{self.model_name.upper()}_{total_timesteps // 1000}k_{self.iter_num}")
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

    