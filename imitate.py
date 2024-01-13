import numpy as np
from stable_baselines3 import PPO

from env import HumanoidEnv


if __name__ == '__main__':
    env = HumanoidEnv(data_path='data/ACCAD', frame_skip=1)

    model = PPO(policy='MlpPolicy', env=env, verbose=1)
    model.learn(total_timesteps=10_000)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()

    env.close()
