from expert_demo import bc_from_demo_v2, rnd_from_demo, BCCallback, get_train_data
import numpy as np
import gym

from abc import ABC, abstractmethod
from util import cuda_to_np, to_cuda_maybe, np_to_cuda
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

import torch

from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3.common.buffers import ReplayBufferByEpisode

import copy

from dril_pack.dril_reward import get_dril_reward

from js_irl import DrilReward, AltRewardEval

from stable_baselines3.common.noise import NormalActionNoise

from random_wd import RandomWD, WDMetric


@hydra.main(config_path="../config", config_name="dril_td3_hopper.yaml")
def Dril_td3(opt: DictConfig):
    dril, env_info = get_dril_reward(opt)

    train_data = get_train_data(opt)

    alt_reward = DrilReward(dril, env_info, env=None, squash_action=False)

    train_env = VecNormalize(make_vec_env(opt.env_id))
    # train_env.obs_rms = train_data.normalizer.obs_rms
    eval_env = VecNormalize(make_vec_env(opt.env_id, n_envs=opt.n_envs), training=False)

    alt_eval_cb = AltRewardEval(alt_reward, eval_env)

    act_noise = NormalActionNoise(mean=np.zeros(eval_env.action_space.shape), sigma=opt.noise_std * np.ones(eval_env.action_space.shape))

    policy_kwargs = eval(opt.policy_kwargs)
    # sac = TD3("MlpPolicy", train_env, action_noise=act_noise, learning_rate=opt.learning_rate,
    #           buffer_size=opt.buffer_size, learning_starts=opt.learning_start, policy_kwargs=policy_kwargs,
    #           gamma=opt.gamma)

    replay_args = {
        "n_step": 1,
        "gamma": opt.gamma,
    }

    wd_model = RandomWD(128, eval_env.observation_space, eval_env.action_space, gamma=0.01)
    wd_model.cuda()
    wd_metric = WDMetric((train_data.observations, train_data.actions), train_env, wd_model)

    td3 = TD3("MlpPolicy", train_env, action_noise=act_noise, policy_kwargs=policy_kwargs, buffer_size=opt.buffer_size,
              # replay_buffer_class=ReplayBufferByEpisode, replay_buffer_kwargs=replay_args,
              learning_rate=opt.learning_rate,
              gamma=opt.gamma,
              learning_starts=opt.learning_start,
              alt_reward=alt_reward,
              wd_metric=wd_metric)
    # alt_reward = alt_reward

    model_name = f"dril_td3_{opt.env_id}_{opt.trial}"

    eval_cb = EvalCallback(eval_env=eval_env, callback_on_new_best=None, n_eval_episodes=opt.n_evals,
                           eval_freq=opt.eval_freq,
                           log_path=None, best_model_save_path="./saves", best_name=model_name)

    eval_cb._log_success_callback = alt_eval_cb

    train_data.normalizer = train_env
    bc_cb = BCCallback(train_data, td3.policy, batch_size=opt.batch_size, epochs=1, lr=2.5e-4)

    # warm_start = BCCallback(train_data, sac.policy, batch_size=128, epochs=200, lr=2.5e-4)
    #
    # warm_start.after_train_step()

    td3.learn(total_timesteps=opt.train_steps, callback=[eval_cb])


@hydra.main(config_path="../config", config_name="dril_td3_hopper.yaml")
def test_td3_bc(opt: DictConfig):
    dril, env_info = get_dril_reward(opt)

    train_data = get_train_data(opt)

    alt_reward = DrilReward(dril, env_info, env=None, squash_action=False)

    train_env = VecNormalize(make_vec_env(opt.env_id))
    eval_env = VecNormalize(make_vec_env(opt.env_id, n_envs=opt.n_envs), training=False)
    eval_env.obs_rms = train_data.normalizer.obs_rms

    alt_eval_cb = AltRewardEval(alt_reward, eval_env)

    # act_noise = NormalActionNoise(mean=np.zeros(eval_env.action_space.shape), sigma=opt.noise_std * np.ones(eval_env.action_space.shape))
    act_noise = None

    policy_kwargs = eval(opt.policy_kwargs)
    sac = TD3("MlpPolicy", train_env, action_noise=act_noise, batch_size=opt.rl_batch_size, learning_rate=opt.learning_rate,
              buffer_size=opt.buffer_size, learning_starts=opt.learning_start, policy_kwargs=policy_kwargs,
              gamma=opt.gamma, alt_reward=alt_reward)
    # alt_reward = alt_reward

    model_name = f"dril_td3_{opt.env_id}_{opt.trial}"

    warm_start = BCCallback(train_data, sac.policy, batch_size=128, epochs=200, lr=2.5e-4)

    warm_start.after_train_step()

    real_rews, ep_len, alt_rews = evaluate_policy(sac, eval_env, n_eval_episodes=20, callback=alt_eval_cb,
                                                  return_episode_rewards=True)
    print(list(zip(real_rews, alt_rews)))


if __name__ == '__main__':
    Dril_td3()
    # test_td3_bc()