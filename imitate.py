from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from env import HumanoidEnv
from agent import ActorCriticPolicy


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        self.logger.record('reward/pose', self.locals['infos'][0]['reward_pose'])
        self.logger.record('reward/velocity', self.locals['infos'][0]['reward_vel'])
        self.logger.record('reward/end', self.locals['infos'][0]['reward_end'])
        self.logger.record('reward/center', self.locals['infos'][0]['reward_center'])

        return True


if __name__ == '__main__':
    env = HumanoidEnv(data_path='data/ACCAD', frame_skip=1, render_mode="human")

    from gymnasium.utils.env_checker import check_env
    check_env(env.unwrapped)
    from stable_baselines3.common.env_checker import check_env
    check_env(env)

    model = PPO(policy=ActorCriticPolicy, env=env, n_steps=2**14, batch_size=256, n_epochs=8,
                gamma=0.95, tensorboard_log='./log/ppo_imitate_tensorboard', verbose=1)
    model.learn(total_timesteps=100_000, callback=TensorboardCallback(), progress_bar=True)
    model.save('agent_imitate')

    model = PPO.load('agent_imitate')

    vec_env = model.get_env() if model.get_env() else make_vec_env(lambda: env)
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render('human')

    env.close()
