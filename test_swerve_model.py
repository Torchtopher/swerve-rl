import swerve_env
import gymnasium as gym 


import gymnasium as gym

from sbx import SAC

# load trained model 
model = SAC.load("swervestatic10000000")

env = gym.make("SwerveEnv-v0", render_mode="human")
#env = gym.make("MountainCarContinuous-v0", render_mode="human")

obs, _  = env.reset()
n_steps = 1000
total_reward = 0
for i in range(10000):
  print("Step:", i)
  should_render = True 
  for _ in range(500):
      action, _state = model.predict(obs, deterministic=True)
      obs, reward, done, trunc, info = env.step(action)
      if reward > 0:
        print("Reward:", reward)
        should_render = True 
      reward += reward
      env.render() if should_render else None 
      if done or trunc:
          #print("Done, reward:", reward/(_+1))
          obs, _ = env.reset()
          break
    