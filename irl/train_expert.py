import hydra
from omegaconf import DictConfig, OmegaConf

import gym

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3.common.noise import NormalActionNoise

from stable_baselines3.common.buffers import ReplayBufferByEpisode

import numpy as np


def train_expert(opt: DictConfig):
    train_env = VecNormalize(make_vec_env(opt.env_id, n_envs=opt.n_envs))
    eval_env = VecNormalize(make_vec_env(opt.env_id, n_envs=opt.n_envs))

    policy_kwargs = eval(opt.policy_kwargs)
    sac = SAC("MlpPolicy", train_env, batch_size=opt.batch_size, learning_rate=opt.learning_rate,
              buffer_size=opt.buffer_size, learning_starts=opt.learning_start, policy_kwargs=policy_kwargs,
              gamma=opt.gamma, tau=opt.tau, gradient_steps=opt.gradient_steps, train_freq=opt.train_freq)

    model_name = f"sac_{opt.env_id}_{opt.trial}"

    eval_cb = EvalCallback(eval_env=eval_env, callback_on_new_best=None, n_eval_episodes=opt.n_evals,
                           eval_freq=opt.eval_freq,
                           log_path="./logs", best_model_save_path="./saves", best_name=model_name)

    sac.learn(total_timesteps=opt.train_steps, callback=eval_cb)

def td3_expert(opt: DictConfig):
    train_env = VecNormalize(make_vec_env(opt.env_id))
    eval_env = VecNormalize(make_vec_env(opt.env_id, n_envs=opt.n_envs), training=False)

    act_noise = NormalActionNoise(mean=np.zeros(eval_env.action_space.shape),
                                  sigma=opt.noise_std * np.ones(eval_env.action_space.shape))

    policy_kwargs = eval(opt.policy_kwargs)
    # td3 = TD3("MlpPolicy", train_env, action_noise=act_noise, learning_rate=opt.learning_rate,
    #           buffer_size=opt.buffer_size, learning_starts=opt.learning_start, policy_kwargs=policy_kwargs,
    #           gamma=opt.gamma)



    td3 = TD3("MlpPolicy", train_env, action_noise=act_noise, policy_kwargs=policy_kwargs, learning_starts=opt.learning_start)

    model_name = f"td3_{opt.env_id}_{opt.trial}"

    eval_cb = EvalCallback(eval_env=eval_env, callback_on_new_best=None, n_eval_episodes=opt.n_evals,
                           eval_freq=opt.eval_freq,
                           log_path="./logs", best_model_save_path="./saves", best_name=model_name)

    td3.learn(total_timesteps=opt.train_steps, callback=eval_cb)


def td3_expert_custom_replay(opt: DictConfig):
    train_env = VecNormalize(make_vec_env(opt.env_id))
    eval_env = VecNormalize(make_vec_env(opt.env_id, n_envs=opt.n_envs), training=False)

    act_noise = NormalActionNoise(mean=np.zeros(eval_env.action_space.shape),
                                  sigma=opt.noise_std * np.ones(eval_env.action_space.shape))

    policy_kwargs = eval(opt.policy_kwargs)
    # td3 = TD3("MlpPolicy", train_env, action_noise=act_noise, learning_rate=opt.learning_rate,
    #           buffer_size=opt.buffer_size, learning_starts=opt.learning_start, policy_kwargs=policy_kwargs,
    #           gamma=opt.gamma)

    replay_args = {
        "n_step": 3,
        "gamma": opt.gamma,
    }
    print("custom replay")
    print(replay_args)
    print(opt)
    td3 = TD3("MlpPolicy", train_env, action_noise=act_noise, policy_kwargs=policy_kwargs,
              replay_buffer_class=ReplayBufferByEpisode, replay_buffer_kwargs=replay_args, learning_starts=opt.learning_start)

    model_name = f"td3_{opt.env_id}_{opt.trial}"

    eval_cb = EvalCallback(eval_env=eval_env, callback_on_new_best=None, n_eval_episodes=opt.n_evals,
                           eval_freq=opt.eval_freq,
                           log_path="./logs", best_model_save_path="./saves", best_name=model_name)

    td3.learn(total_timesteps=opt.train_steps, callback=eval_cb)


@hydra.main(config_path="../config", config_name="sac_rl_hopper.yaml")
def rl_hopper(opt: DictConfig):
    train_expert(opt)


@hydra.main(config_path="../config", config_name="sac_rl_reacher.yaml")
def rl_reacher(opt: DictConfig):
    train_expert(opt)


@hydra.main(config_path="../config", config_name="dril_td3_hopper.yaml")
def td3_hopper(opt: DictConfig):
    # td3_expert(opt)
    td3_expert_custom_replay(opt)


if __name__ == '__main__':
    # rl_hopper()
    # rl_reacher()
    td3_hopper()