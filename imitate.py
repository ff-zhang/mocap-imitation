from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from env import HumanoidEnv


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record('pose reward', self.training_env.get_attr('reward_pose'))

        return True


if __name__ == '__main__':
    env = HumanoidEnv(data_path='data/ACCAD', frame_skip=1)

    from gymnasium.utils.env_checker import check_env
    check_env(env.unwrapped)
    from stable_baselines3.common.env_checker import check_env
    check_env(env)

    model = PPO(policy='MultiInputPolicy', env=env, verbose=1, tensorboard_log='./log/ppo_imitate_tensorboard')
    model.learn(total_timesteps=10_000)

    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()

    env.close()
