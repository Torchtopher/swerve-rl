import swerve_env
import gymnasium as gym 
import cv2
import keyboard
import gymnasium as gym
#from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ
import numpy as np
from stable_baselines3.common.env_checker import check_env
import time

env = gym.make("SwerveEnv-v0", render_mode="human")
res = check_env(env)

obs = env.reset()
n_steps = 1000
action = [1, 1, 0]

for _ in range(n_steps):

    obs, reward, done, trunc, info = env.step(action)
    img = env.render()
    if done or trunc:
        print("done") if done else print("trunc")
        #obs = env.reset()
        # env.render until ctrl+c
        try:
            while True:
                env.render()
        except KeyboardInterrupt:
            break
