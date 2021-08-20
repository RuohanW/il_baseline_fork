from expert_demo import bc_from_demo_v2, rnd_from_demo, BCCallback, get_train_data
import numpy as np
import gym

from abc import ABC, abstractmethod
from util import cuda_to_np, to_cuda_maybe, np_to_cuda
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf

import torch

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from stable_baselines3.common.callbacks import EvalCallback

import copy

from dril_pack.dril_reward import get_dril_reward


class Reward(ABC):
    def __init__(self, reward_model, env=None, squash_action=True):
        self.reward_model = reward_model
        self.env = env
        self.squash_action = squash_action

        self.rew_clip = 10

    @abstractmethod
    def _reward_forward(self, obs, actions):
        """
        Get reward from the reward_model
        """

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.env.action_space.low, self.env.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = self.env.action_space.low, self.env.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def __call__(self, obs, actions):
        if self.env:
            obs = self.env.normalize_obs(obs)

        if self.squash_action and isinstance(self.env.action_space, gym.spaces.Box):
            actions = self.unscale_action(actions)

        res = []

        batch_size = 1024
        if obs.shape[0] < batch_size:
            batch_size = obs.shape[0]
        iter_size = obs.shape[0]//batch_size
        for i in range(iter_size):
            batch_obs = obs[i*batch_size:(i+1)*batch_size]
            batch_act = actions[i*batch_size:(i+1)*batch_size]
            res.append(self._reward_forward(batch_obs, batch_act))

        if iter_size*batch_size < obs.shape[0]:
            res.append(self._reward_forward(np_to_cuda(obs[batch_size*iter_size:]), np_to_cuda(actions[batch_size*iter_size:])))

        rews = np.concatenate(res)
        rews = np.clip(rews, -self.rew_clip, self.rew_clip)

        return rews


class ActionProbReward(Reward):
    def __init__(self, reward_model, env=None, squash_action=True):
        super(ActionProbReward, self).__init__(reward_model, env, squash_action)

    @torch.no_grad()
    def _reward_forward(self, obs, actions):
        # mean_actions, log_std, _ = self.reward_model.get_action_dist_params(obs)
        # dist = self.reward_model.action_dist.proba_distribution(mean_actions, log_std)
        # log_prob = dist.log_prob(actions)
        # return cuda_to_np(log_prob, dtype=np.float32)
        obs = np_to_cuda(obs)
        actions= np_to_cuda(actions)

        pred_actions = self.reward_model._predict(obs, deterministic=True)
        log_prob = -torch.square(pred_actions-actions).sum(dim=1)
        return cuda_to_np(log_prob, dtype=np.float32)


class DrilReward(Reward):
    def __init__(self, reward_model, env_info, env=None, squash_action=True):
        super(DrilReward, self).__init__(reward_model, env, squash_action)
        self.env_info = env_info

    @torch.no_grad()
    def _reward_forward(self, obs, actions):
        # mean_actions, log_std, _ = self.reward_model.get_action_dist_params(obs)
        # dist = self.reward_model.action_dist.proba_distribution(mean_actions, log_std)
        # log_prob = dist.log_prob(actions)
        # return cuda_to_np(log_prob, dtype=np.float32)
        obs = np_to_cuda(obs)
        actions = np_to_cuda(actions)

        rews = self.reward_model.predict_reward(actions, obs, self.env_info)
        return rews

class RNDReward(Reward):
    def __init__(self, reward_model, env=None, squash_action=True):
        super(RNDReward, self).__init__(reward_model, env, squash_action)

    def __call__(self, obs, actions):
        if self.squash_action and isinstance(self.env.action_space, gym.spaces.Box):
            actions = self.unscale_action(actions)

        t_in = np.concatenate([obs, actions], axis=1)

        if self.env:
            obs = self.env.normalize_obs(t_in)

        res = []

        batch_size = 1024
        if obs.shape[0] < batch_size:
            batch_size = obs.shape[0]
        iter_size = obs.shape[0]//batch_size
        for i in range(iter_size):
            batch_obs = t_in[i*batch_size:(i+1)*batch_size]
            obs_cuda = np_to_cuda(batch_obs)
            res.append(self._reward_forward(obs_cuda))

        if iter_size*batch_size < obs.shape[0]:
            res.append(self._reward_forward(np_to_cuda(t_in[batch_size*iter_size:])))

        rews = np.concatenate(res)

        return rews

    @torch.no_grad()
    def _reward_forward(self, t_in):
        # mean_actions, log_std, _ = self.reward_model.get_action_dist_params(obs)
        # dist = self.reward_model.action_dist.proba_distribution(mean_actions, log_std)
        # log_prob = dist.log_prob(actions)
        # return cuda_to_np(log_prob, dtype=np.float32)

        rews = self.reward_model.reward(t_in)
        return cuda_to_np(rews, dtype=np.float32)


class AltRewardEval(ABC):
    def __init__(self, reward_model, env):
        self.reward_model = reward_model
        self.env = env
        self.actions = []
        self.observations = []
        self.episode_lengths = []
        self.episode_rewards = []
        self.done_envs = []

        self.trajectories = None

    def __call__(self, locals, globals):
        self.actions.append(copy.deepcopy(locals["actions"]))
        self.observations.append(copy.deepcopy(locals["cur_observations"]))

    def on_done(self, locals, globals):
        self.actions = np.stack(self.actions, axis=1) #n_envs X traj_lengths X action_dims
        self.observations = np.stack(self.observations, axis=1)

        self.episode_lengths = locals["episode_lengths"]
        self.episode_rewards = locals["episode_rewards"]
        self.done_envs = locals["done_envs"]

        alt_rewards = self.eval_trajs()
        self.reset()
        return alt_rewards

    def reset(self):
        self.actions = []
        self.observations = []
        self.episode_lengths = []
        self.episode_rewards = []
        self.done_envs = []

    def eval_trajs(self):
        traj_alt_rewards = []
        env_offsets = np.zeros(self.actions.shape[0], dtype=np.int)
        for length, env_id in zip(self.episode_lengths, self.done_envs):
            offset = env_offsets[env_id]
            actions = self.actions[env_id, offset:offset+length]
            observations = self.observations[env_id, offset:offset+length]

            observations = self.env.unnormalize_obs(observations)

            traj_alt_rewards.append(self.get_traj_reward(observations, actions))
            env_offsets[env_id] += length

        return traj_alt_rewards

    def get_traj_reward(self, obs, actions):
        tmp = self.reward_model(obs, actions)
        return np.sum(tmp)

@hydra.main(config_path="../config", config_name="bc_reacher.yaml")
def test_reward_obj(opt: DictConfig):
    actor, env, valid_loader = bc_from_demo_v2(opt)
    alt_reward = ActionProbReward(actor, env=None, squash_action=False)

    for i in range(10):
        observations, actions = valid_loader.dataset.__getitem__(i)
        observations = np.expand_dims(observations, 0)
        actions = np.expand_dims(actions, 0)
        print(alt_reward(observations, actions))

@hydra.main(config_path="../config", config_name="js_irl_hopper.yaml")
def JS_irl(opt: DictConfig):
    actor, env, train_data = bc_from_demo_v2(opt)

    alt_reward = ActionProbReward(actor, env=env, squash_action=False)

    train_env = VecNormalize(make_vec_env(opt.env_id))
    eval_env = VecNormalize(make_vec_env(opt.env_id, n_envs=opt.n_envs), training=False)

    train_data.normalizer = train_env
    alt_eval_cb = AltRewardEval(alt_reward, eval_env)

    policy_kwargs = eval(opt.policy_kwargs)
    sac = SAC("MlpPolicy", train_env, batch_size=opt.rl_batch_size, learning_rate=opt.learning_rate,
              buffer_size=opt.buffer_size, learning_starts=opt.learning_start, policy_kwargs=policy_kwargs, ent_coef=1.,
              gamma=opt.gamma, tau=opt.tau, gradient_steps=opt.gradient_steps, train_freq=opt.train_freq, alt_reward=alt_reward)


    model_name = f"irl_sac_{opt.env_id}_{opt.trial}"

    eval_cb = EvalCallback(eval_env=eval_env, callback_on_new_best=None, n_eval_episodes=opt.n_evals,
                           eval_freq=opt.eval_freq,
                           log_path=None, best_model_save_path="./saves", best_name=model_name)

    eval_cb._log_success_callback = alt_eval_cb

    bc_cb = BCCallback(train_data, sac.policy, batch_size=opt.batch_size, lr=2.5e-4)

    sac.learn(total_timesteps=opt.train_steps, callback=[eval_cb])


@hydra.main(config_path="../config", config_name="js_irl_hopper.yaml")
def RND_irl(opt: DictConfig):
    rnd, env, train_data, obs_rms, model = rnd_from_demo(opt)

    alt_reward = RNDReward(rnd, env, squash_action=False)

    train_env = VecNormalize(make_vec_env(opt.env_id))
    train_env.obs_rms = obs_rms
    eval_env = VecNormalize(make_vec_env(opt.env_id, n_envs=opt.n_envs), training=False)

    alt_eval_cb = AltRewardEval(alt_reward, eval_env)

    policy_kwargs = eval(opt.policy_kwargs)
    sac = SAC("MlpPolicy", train_env, batch_size=opt.rl_batch_size, learning_rate=opt.learning_rate,
              buffer_size=opt.buffer_size, learning_starts=opt.learning_start, policy_kwargs=policy_kwargs,
              gamma=opt.gamma, tau=opt.tau, gradient_steps=opt.gradient_steps, train_freq=opt.train_freq, alt_reward=alt_reward)
    # alt_reward = alt_reward

    model_name = f"irl_sac_{opt.env_id}_{opt.trial}"

    eval_cb = EvalCallback(eval_env=eval_env, callback_on_new_best=None, n_eval_episodes=opt.n_evals,
                           eval_freq=opt.eval_freq,
                           log_path=None, best_model_save_path="./saves", best_name=model_name)

    eval_cb._log_success_callback = alt_eval_cb

    train_data.normalizer = train_env
    bc_cb = BCCallback(train_data, sac.policy, batch_size=opt.batch_size, lr=3e-4)

    sac.learn(total_timesteps=opt.train_steps, callback=[eval_cb])


@hydra.main(config_path="../config", config_name="dril_irl_hopper.yaml")
def Dril_irl(opt: DictConfig):
    dril, env_info = get_dril_reward(opt)

    train_data = get_train_data(opt)

    alt_reward = DrilReward(dril, env_info, env=None, squash_action=False)

    train_env = VecNormalize(make_vec_env(opt.env_id))
    eval_env = VecNormalize(make_vec_env(opt.env_id, n_envs=opt.n_envs), training=False)

    alt_eval_cb = AltRewardEval(alt_reward, eval_env)

    policy_kwargs = eval(opt.policy_kwargs)
    sac = SAC("MlpPolicy", train_env, batch_size=opt.rl_batch_size, learning_rate=opt.learning_rate,
              buffer_size=opt.buffer_size, learning_starts=opt.learning_start, policy_kwargs=policy_kwargs, ent_coef=0.1,
              gamma=opt.gamma, tau=opt.tau, gradient_steps=opt.gradient_steps, train_freq=opt.train_freq, alt_reward=alt_reward)
    # alt_reward = alt_reward

    model_name = f"irl_sac_{opt.env_id}_{opt.trial}"

    eval_cb = EvalCallback(eval_env=eval_env, callback_on_new_best=None, n_eval_episodes=opt.n_evals,
                           eval_freq=opt.eval_freq,
                           log_path=None, best_model_save_path="./saves", best_name=model_name)

    eval_cb._log_success_callback = alt_eval_cb

    train_data.normalizer = train_env
    bc_cb = BCCallback(train_data, sac.policy, batch_size=opt.batch_size, epochs=5, lr=2.5e-4, deterministic=False)

    warm_start = BCCallback(train_data, sac.policy, batch_size=128, epochs=200, lr=2.5e-4, deterministic=False)

    warm_start.after_train_step()

    sac.learn(total_timesteps=opt.train_steps, callback=[eval_cb, bc_cb])


@hydra.main(config_path="../config", config_name="js_irl_hopper.yaml")
def test_rnd_reward(opt):
    rnd, env, train_data, obs_rms, model = rnd_from_demo(opt)

    alt_reward = RNDReward(rnd, env, squash_action=False)

    eval_env = VecNormalize(make_vec_env(opt.env_id, n_envs=opt.n_envs), training=False)
    eval_env.obs_rms = obs_rms

    alt_eval_cb = AltRewardEval(alt_reward, eval_env)

    rewards = evaluate_policy(model, eval_env, n_eval_episodes=20, callback=alt_eval_cb, return_episode_rewards=True)
    print(rewards)


@hydra.main(config_path="../config", config_name="dril_irl_hopper.yaml")
def test_dril_reward(opt):
    dril, env_info = get_dril_reward(opt)

    alt_reward = DrilReward(dril, env_info, env=None, squash_action=False)
    model, env, _ = bc_from_demo_v2(opt)

    eval_env = VecNormalize(make_vec_env(opt.env_id, n_envs=opt.n_envs), training=False)
    eval_env.obs_rms = env.obs_rms

    alt_eval_cb = AltRewardEval(alt_reward, eval_env)

    real_rews, ep_len, alt_rews = evaluate_policy(model, eval_env, n_eval_episodes=20, callback=alt_eval_cb, return_episode_rewards=True, deterministic=False)
    print(list(zip(real_rews, alt_rews)))



@hydra.main(config_path="../config", config_name="dril_irl_hopper.yaml")
def Dril_irl(opt: DictConfig):
    dril, env_info = get_dril_reward(opt)

    train_data = get_train_data(opt)

    alt_reward = DrilReward(dril, env_info, env=None, squash_action=False)

    train_env = VecNormalize(make_vec_env(opt.env_id))
    eval_env = VecNormalize(make_vec_env(opt.env_id, n_envs=opt.n_envs), training=False)

    alt_eval_cb = AltRewardEval(alt_reward, eval_env)

    policy_kwargs = eval(opt.policy_kwargs)
    sac = SAC("MlpPolicy", train_env, batch_size=opt.rl_batch_size, learning_rate=opt.learning_rate,
              buffer_size=opt.buffer_size, learning_starts=opt.learning_start, policy_kwargs=policy_kwargs, ent_coef=0.1,
              gamma=opt.gamma, tau=opt.tau, gradient_steps=opt.gradient_steps, train_freq=opt.train_freq, alt_reward=alt_reward)
    # alt_reward = alt_reward

    model_name = f"irl_sac_{opt.env_id}_{opt.trial}"

    eval_cb = EvalCallback(eval_env=eval_env, callback_on_new_best=None, n_eval_episodes=opt.n_evals,
                           eval_freq=opt.eval_freq,
                           log_path=None, best_model_save_path="./saves", best_name=model_name)

    eval_cb._log_success_callback = alt_eval_cb

    train_data.normalizer = train_env
    bc_cb = BCCallback(train_data, sac.policy, batch_size=opt.batch_size, epochs=5, lr=2.5e-4, deterministic=False)

    warm_start = BCCallback(train_data, sac.policy, batch_size=128, epochs=200, lr=2.5e-4, deterministic=False)

    warm_start.after_train_step()

    sac.learn(total_timesteps=opt.train_steps, callback=[eval_cb, bc_cb])


if __name__ == '__main__':
    # JS_irl()
    # RND_irl()
    # test_rnd_reward()
    Dril_irl()
    # test_dril_reward()


