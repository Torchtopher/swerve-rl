import swerve_env
import gymnasium as gym 


import gymnasium as gym
from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ
import numpy as np
from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from typing import Callable
from collections import OrderedDict
from inspect import signature

def filter_hyperparams(algo_class, hyperparams):
    algo_signature = signature(algo_class)
    valid_params = {
        key: value for key, value in hyperparams.items()
        if key in algo_signature.parameters
    }
    return valid_params

def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

class VideoRecorderCallback(BaseCallback):
    """
    Custom callback for recording videos at regular intervals.
    """

    def __init__(self, eval_env, video_folder, record_freq, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.video_folder = video_folder
        self.record_freq = record_freq
        self.video_count = 0

    def _on_step(self) -> bool:
        #print(f"Step: {self.n_calls}")
        if self.n_calls % self.record_freq == 0:
            with RecordVideo(self.eval_env, video_folder=self.video_folder, disable_logger=True, name_prefix=f"{self.n_calls}-rl-video") as video_env:
                obs, _ = video_env.reset()
                done = False
                truncated = False
                while not done and not truncated:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, done, truncated, _ = video_env.step(action)

            self.video_count += 1
        return True

TIMESTEPS = 10_000_000 


def main():
  #env_id = "MountainCarContinuous-v0"
  env_id = "SwerveEnv-v0"
  num_cpu = 16  # Number of processes to use
  record_freq = TIMESTEPS // 2 // num_cpu
  log_folder = "./swerve_training_logs/"

  env = SubprocVecEnv([make_env(env_id, i) for i in range(16)])

  vec_env = make_vec_env(env_id, n_envs=num_cpu)

  # model = TD3("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_folder,
  #             buffer_size=1000000,
  #             batch_size=256,
  #             learning_rate=0.003,
  #             learning_starts=10000,
  #             policy_kwargs=dict(net_arch=[400, 300]),
  #             tau=0.01,
  #             train_freq=1,
  #             gradient_steps=1,
  #             )
              
# +


  model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=log_folder, 
              n_steps=1024,
              gamma=0.9999,
              learning_rate=6e-4,
              batch_size=256,
              gae_lambda=0.98,
              )
  # We create a separate environment for evaluation
  eval_env = gym.make(env_id, render_mode="rgb_array")
  video_callback = VideoRecorderCallback(eval_env, "./swerve_training_logs", record_freq)
  # Random Agent, before training
  mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
  print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")


  
  model.learn(callback=video_callback, total_timesteps=TIMESTEPS, progress_bar=True)
  model.save("swervestatic" + str(type(model)) + str(TIMESTEPS))
  #
  #del model # remove to demonstrate saving and loading

  obs, _ = eval_env.reset()
  for i in range(1000):
      action, _state = model.predict(obs, deterministic=False)
      obs, reward, done, trunc, info = eval_env.step(action)
      print("Reward:", reward)
      eval_env.render()
      # VecEnv resets automatically
      if done or trunc:
        obs = eval_env.reset()
  
if __name__ == "__main__":
  main()