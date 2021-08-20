import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import gym

from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

from stable_baselines3.common.vec_env import sync_envs_normalization

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.distributions import DiagGaussianDistribution

from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.nn import functional as F
import torch
from torch import optim

import pickle
from abc import ABC
from util import AverageMeter, to_cuda_maybe, cuda_to_np

from tqdm import tqdm

import copy

import math

class DemoCollector(ABC):
    def __init__(self, save_path):
        self.save_path = save_path
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

    def demo_filter(self, target_length=0, target_reward=0, normalized_env=None):
        trajectories = []
        env_offsets = np.zeros(self.actions.shape[0], dtype=np.int)
        for length, reward, env_id in zip(self.episode_lengths, self.episode_rewards, self.done_envs):
            offset = env_offsets[env_id]
            if length >= target_length and reward >= target_reward:
                actions = self.actions[env_id, offset:offset+length]
                observations = self.observations[env_id, offset:offset+length]

                if normalized_env:
                    observations = normalized_env.unnormalize_obs(observations)

                trajectories.append((observations, actions))
            env_offsets[env_id] += length

        self.trajectories = trajectories

    def save(self):
        with open(self.save_path, "wb") as f:
            pickle.dump(self.trajectories, f)


@hydra.main(config_path="../config", config_name="collect_demo.yaml")
def collect_demo(opt: DictConfig):
    eval_env = VecNormalize(make_vec_env(opt.env_id, n_envs=opt.n_envs), training=False)

    cb = DemoCollector(f"{opt.save_dir}/{opt.demo_name}")

    sac = SAC.load(f"{opt.save_dir}/{opt.save_name}")
    sync_envs_normalization(sac.env, eval_env)
    avg_reward, std = evaluate_policy(sac, eval_env, opt.n_evals, callback=cb)

    cb.demo_filter(target_length=1000, normalized_env=eval_env)
    print(len(cb.trajectories))
    cb.save()


class BC_Dataset(Dataset):
    def __init__(self, data, normalizer=None):
        self.observations, self.actions = data
        self.normalizer = normalizer

    def __getitem__(self, item):
        if self.normalizer:
            obs = self.normalizer.normalize_obs(self.observations[item])
        else:
            obs = self.observations[item]
        return obs, self.actions[item]

    def __len__(self):
        return self.observations.shape[0]


class RND_Dataset(Dataset):
    def __init__(self, data, normalizer=None):
        self.data = data
        self.normalizer = normalizer

    def __getitem__(self, item):
        if self.normalizer:
            obs = self.normalizer.normalize_obs(self.data[item])
        else:
            obs = self.data[item]
        return obs

    def __len__(self):
        return self.data.shape[0]


class BC_Data(ABC):
    def __init__(self, split="random", folds=5, trajectories=[], save_path=None, num_traj=None):
        assert type(folds) == int and folds > 1
        if save_path and not trajectories:
            with open(save_path, "rb") as f:
                trajectories = pickle.load(f)
        assert trajectories

        if num_traj:
            trajectories = trajectories[:num_traj]

        print(len(trajectories))

        self.trajectories = trajectories
        self.split = split
        self.folds = folds

        observations, actions = zip(*trajectories)
        observations = np.concatenate(observations)
        actions = np.concatenate(actions)

        self.observations = observations
        self.actions = actions

        ob_rms = RunningMeanStd(self.observations.shape[1])

        ob_rms.mean = np.mean(self.observations, axis=0)
        ob_rms.var = np.var(self.observations, axis=0)

        self.vec_normalize = VecNormalize(make_vec_env("CartPole-v1"))
        self.vec_normalize.obs_rms = ob_rms


    def get_slice(self, i):
        # assert self.folds > i
        # if self.split == "random":
        #     fold_size = self.observations.shape[0]//self.folds
        #     flags = np.zeros(self.observations.shape[0], dtype=np.int)
        #     flags[fold_size*i:fold_size*(i+1)] = 1
        #     valid_inds = self.shuffle[flags.astype(np.bool)]
        #     train_inds = self.shuffle[(1-flags).astype(np.bool)]
        #     valid_observations = self.observations[valid_inds]
        #     valid_actions = self.actions[valid_inds]
        #
        #     train_observations = self.observations[train_inds]
        #     train_actions = self.actions[train_inds]
        # elif self.split == "traj":
        valid_actions = []
        valid_observations = []

        train_actions = []
        train_observations = []
        traj_size = len(self.trajectories)//self.folds

        for j, (observations, actions) in enumerate(self.trajectories):
            if (traj_size*i <= j) and (j < traj_size*(i+1)):
                valid_actions.append(actions)
                valid_observations.append(observations)
            else:
                train_actions.append(actions)
                train_observations.append(observations)

        valid_actions = np.concatenate(valid_actions)
        valid_observations = np.concatenate(valid_observations)

        train_observations = np.concatenate(train_observations)
        train_actions = np.concatenate(train_actions)

        return BC_Dataset((train_observations, train_actions), normalizer=self.vec_normalize), BC_Dataset((valid_observations, valid_actions), normalizer=self.vec_normalize)

    # def set_mode(self, mode):
    #     if mode == "train":
    #         self.cur_observations = self.normalize_obs(self.train_observations)
    #         self.cur_actions = self.train_actions
    #     elif mode == "valid":
    #         self.cur_observations = self.normalize_obs(self.valid_observations)
    #         self.cur_actions = self.valid_actions
    #     else:
    #         raise NotImplementedError("Only accept 'train' or 'valid' mode")

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return self.vec_normalize.normalize_obs(obs)


class RND(ABC):
    def __init__(self, in_dim, normalizer, out_dim=128, target_arch=[256, 256], pred_arch=[256, 256], threshold=0.95):
        super(RND, self).__init__()

        self.target_model = self._construct_model("target", in_dim, target_arch, out_dim)
        self.pred_model = self._construct_model("pred", in_dim, pred_arch, out_dim)
        self.alpha = 1
        self.log_thres = -np.log(threshold)
        self.normalizer = normalizer

    def train(self, data_loader, epochs, lr):
        optimizer = torch.optim.Adam(self.pred_model.parameters(), lr=lr)

        for _ in range(epochs):
            epoch_loss = 0
            for t_in in data_loader:
                t_in = t_in.cuda()
                optimizer.zero_grad()
                target = self.target_model(t_in)
                pred = self.pred_model(t_in)

                m_loss = F.mse_loss(pred, target)
                epoch_loss += m_loss.item()

                m_loss.backward()
                optimizer.step()

            self.set_alpha(data_loader)

    def set_alpha(self, data_loader):
        t_max = 0
        epoch_loss = 0
        for t_in in data_loader:
            t_in = t_in.cuda()

            err = self.model_loss(t_in)

            err_max = np.max(cuda_to_np(err, np.float32))
            t_max = max(t_max, err_max)
            epoch_loss += err.sum().item()

        self.alpha = self.log_thres / t_max
        # self.alpha = t_max + 1e-8

        print(epoch_loss)
        print(t_max)
        print(self.alpha)
        print("-----------------")


    def model_loss(self, data):
        target = self.target_model(data)
        pred = self.pred_model(data)
        err = torch.square(target - pred).sum(dim=1)

        return err

    def reward(self, data):
        err = self.model_loss(data)
        # return (err < self.alpha).double()
        return torch.exp(-self.alpha*err)

    @staticmethod
    def _construct_model(name, in_dim, arch, out_dim):
        model = torch.nn.Sequential()
        tmp = [in_dim] + arch
        for i in range(len(tmp)-1):
            model.add_module(f"{name}_fc_{i}", torch.nn.Linear(tmp[i], tmp[i+1]))
            model.add_module(f"relu_{i}", torch.nn.ReLU())
        model.add_module(f"{name}_fc_out", torch.nn.Linear(tmp[-1], out_dim))
        return model.cuda()


class BCCallback(BaseCallback):
    def __init__(self, train_data, policy, batch_size=256, lr=3e-4, epochs=1, num_steps=None, deterministic=False, verbose=True):
        super(BCCallback, self).__init__(verbose)
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
        self.epochs = epochs
        self.policy = policy
        self.lr = lr
        self.num_steps = num_steps
        self.deterministic = deterministic

        #Assume that actor and critic does not share parameters
        self.optimizer = optim.Adam(self.policy.actor.parameters(), lr=lr)

    def _after_train_step(self) -> None:
        train_loss = 0
        counter = 0
        for _ in range(self.epochs):
            for i, (observations, actions) in enumerate(self.train_loader):
                # if self.num_steps is None or i < self.num_steps:
                self.optimizer.zero_grad()
                observations = to_cuda_maybe(observations)
                actions = to_cuda_maybe(actions)
                pred_actions = self.policy.actor._predict(observations, deterministic=self.deterministic)
                # pred_actions = torch.clip(pred_actions, -1, 1)
                m_loss = F.mse_loss(pred_actions, actions)

                train_loss += m_loss.item()
                counter += 1

                m_loss.backward()
                self.optimizer.step()
        print(train_loss/counter)

    def _on_step(self) -> bool:
        return True


def eval_bc(data_loader, actor):
    log_probs = []
    for _, (observations, actions) in enumerate(tqdm(data_loader)):
        observations = to_cuda_maybe(observations)
        actions = to_cuda_maybe(actions)

        pred_actions = actor._predict(observations, deterministic=True)

        m_loss = torch.mean(torch.square((actions - pred_actions)))
        log_probs.append(-m_loss.item())

        # mean_actions, log_std, _ = actor.get_action_dist_params(observations)
        # dist = actor.action_dist.proba_distribution(mean_actions, log_std)
        # log_prob = dist.log_prob(actions)
        # log_probs.append(cuda_to_np(log_prob, dtype=np.float32))

        # prob_dist = gaussian.proba_distribution(mean_actions, log_std)
        # k_loss = torch.mean(prob_dist.log_prob(actions))
        # k_losses.append(k_loss.item())
    # log_probs = np.concatenate(log_probs)
    log_probs = np.asarray(log_probs)
    return -np.mean(log_probs)


def save_actor(actor, save_path):
    torch.save(actor.state_dict(), save_path)


def load_actor(actor, save_path):
    with open(save_path, "rb") as f:
        state_dict = torch.load(f)
        actor.load_state_dict(state_dict)


@hydra.main(config_path="../config", config_name="collect_demo.yaml")
def bc_from_demo(opt: DictConfig):
    save_path = f"{opt.save_dir}/{opt.demo_name}"

    data = BC_Data(save_path=save_path, num_traj=5)

    train_db, valid_db = data.get_slice(0)
    train_loader = DataLoader(train_db, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=False)
    valid_loader = DataLoader(valid_db, batch_size=opt.batch_size*2, shuffle=False, num_workers=4, drop_last=False)

    policy_kwargs = dict(net_arch=[400, 300])
    env = make_vec_env(opt.env_id)
    sac = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs)
    actor = sac.policy.actor
    # actor.action_dist = DiagGaussianDistribution(env.action_space.shape)

    optimizer = optim.Adam(actor.parameters(), lr=opt.lr)


    best = 1000
    avg_meter = AverageMeter()
    action_meter = AverageMeter()

    for _ in range(opt.epochs):
        for i, (observations, actions) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            observations = to_cuda_maybe(observations)
            actions = to_cuda_maybe(actions)
            pred_actions = actor._predict(observations, deterministic=True)
            m_loss = F.mse_loss(pred_actions, actions)

            # mean_actions, log_std, _ = actor.get_action_dist_params(observations)
            # std = torch.exp(log_std)
            # m_loss = torch.mean(torch.sum(0.5*torch.square((actions - mean_actions)/std) + log_std, dim=1))
            #
            # action_loss = F.mse_loss(mean_actions, actions)

            m_loss.backward()
            optimizer.step()
            avg_meter.update(m_loss.item())
            # action_meter.update(action_loss.item())

        print(avg_meter.avg)
        # # print(action_meter.avg)
        # avg_meter.reset()
        # action_meter.reset()

        # log_prob = eval_bc(valid_loader, actor)
        # print(f"log_prob: {log_prob}")
        # if log_prob < best:
        #     best = log_prob
        #     print(f"best:{log_prob}")
        #     save_actor(actor, f"{opt.save_dir}/{opt.env_id}_actor")


    # load_actor(actor, f"{opt.save_dir}/{opt.env_id}_actor")

    eval_env = VecNormalize(make_vec_env(opt.env_id, n_envs=opt.n_envs), training=False)

    sync_envs_normalization(data.vec_normalize, eval_env)
    print(eval_env.obs_rms.mean)
    # old = SAC.load(f"{opt.save_dir}/{opt.save_name}")
    # print(old.env.obs_rms.mean)
    # sync_envs_normalization(old.env, eval_env)
    avg_reward, std = evaluate_policy(sac, eval_env, opt.n_evals)
    print(avg_reward, std)

@hydra.main(config_path="../config", config_name="collect_demo.yaml")
def bc_sanity_check(opt: DictConfig):
    save_path = f"{opt.save_dir}/{opt.demo_name}"

    data = BC_Data(save_path=save_path)

    train_db, valid_db = data.get_slice(0)
    sac = SAC.load(f"{opt.save_dir}/{opt.save_name}")
    actor = sac.policy.actor

    train_db.normalizer = sac.env
    train_loader = DataLoader(train_db, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=False)

    for _ in range(opt.epochs):
        for i, (observations, actions) in enumerate(tqdm(train_loader)):
            observations = to_cuda_maybe(observations)
            actions = to_cuda_maybe(actions)
            pred_actions = actor._predict(observations)
            m_loss = F.mse_loss(pred_actions, actions)

            print(m_loss.item())

def scale_action(actions, action_space):
    low, high = action_space.low, action_space.high
    return 2.0 * ((actions - low) / (high - low)) - 1.0

def bc_from_demo_v2(opt: DictConfig):
    with open(f"/home/ruohan/projects/baselines/data/{opt.env_id[:-3]}-v2.pkl", "rb") as f:
        state_dict = pickle.load(f)

    traj_length = opt.traj_length
    obs, actions = state_dict["observations"], np.squeeze(state_dict["actions"])

    valid_obs = None

    env = VecNormalize(make_vec_env(opt.env_id, n_envs=opt.n_envs), training=False)
    # actions = scale_action(actions, env.action_space)
    print("Actions clipped")
    actions = np.clip(actions, env.action_space.low, env.action_space.high)

    obs_mean = np.mean(obs, axis=0)
    obs_var = np.var(obs, axis=0)

    demo_len = opt.n_trajs * traj_length
    if demo_len < obs.shape[0]:
        valid_obs = obs[demo_len:]
        valid_actions = actions[demo_len:]

        obs = obs[:demo_len]
        actions = actions[:demo_len]

    rms = RunningMeanStd(shape=env.observation_space.shape)
    rms.mean = obs_mean
    rms.var = obs_var
    env.obs_rms = rms

    valid_loader = None

    train_db = BC_Dataset((obs, actions), normalizer=env)
    train_loader = DataLoader(train_db, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=False)

    if valid_obs is not None:
        valid_db = BC_Dataset((valid_obs,valid_actions), normalizer=env)
        valid_loader = DataLoader(valid_db, batch_size=opt.batch_size*2, shuffle=False, num_workers=4, drop_last=False)

    policy_kwargs = dict(net_arch=[256, 256])

    dummy_env = VecNormalize(make_vec_env(opt.env_id))
    sac = SAC("MlpPolicy", dummy_env, policy_kwargs=policy_kwargs)
    actor = sac.policy.actor
    # actor.action_dist = DiagGaussianDistribution(env.action_space.shape)

    optimizer = optim.Adam(actor.parameters(), lr=opt.lr)

    avg_meter = AverageMeter()

    best = 100

    for _ in range(opt.epochs):
        for i, (observations, actions) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            observations = to_cuda_maybe(observations)
            actions = to_cuda_maybe(actions)
            pred_actions = actor._predict(observations, deterministic=False)
            m_loss = F.mse_loss(pred_actions, actions)

            # mean_actions, log_std, _ = actor.get_action_dist_params(observations)
            # std = torch.exp(log_std)
            # m_loss = torch.mean(torch.sum(0.5*torch.square((actions - mean_actions)/std) + log_std, dim=1))
            #
            # action_loss = F.mse_loss(mean_actions, actions)

            m_loss.backward()
            optimizer.step()
            avg_meter.update(m_loss.item())
            # action_meter.update(action_loss.item())

        print(avg_meter.avg)
        avg_meter.reset()

        if valid_loader is not None:
            log_prob = eval_bc(valid_loader, actor)
            print(f"log_prob: {log_prob}")
            if log_prob < best:
                best = log_prob
                print(f"best:{log_prob}")
                save_actor(actor, f"{opt.save_dir}/{opt.env_id}_actor")

    if valid_loader is not None:
        load_actor(actor, f"{opt.save_dir}/{opt.env_id}_actor")

    env.training = False
    avg_reward, std = evaluate_policy(sac, env, opt.n_evals)
    print(f"{avg_reward:.1f}, {std:.1f}")

    return actor, env, train_db
    # return sac


def rnd_from_demo(opt: DictConfig):
    with open(f"/home/ruohan/projects/baselines/data/{opt.env_id[:-3]}-v2.pkl", "rb") as f:
        state_dict = pickle.load(f)

    traj_length = opt.traj_length
    obs, actions = state_dict["observations"], np.squeeze(state_dict["actions"])

    valid_obs = None

    env = VecNormalize(make_vec_env(opt.env_id, n_envs=opt.n_envs), training=False)
    # actions = scale_action(actions, env.action_space)
    print("Actions clipped")
    actions = np.clip(actions, env.action_space.low, env.action_space.high)

    joint_data = np.concatenate([obs, actions], axis=1)

    obs_mean = np.mean(joint_data, axis=0)
    obs_var = np.var(joint_data, axis=0)

    valid_joint_data = None

    demo_len = opt.n_trajs * traj_length
    if demo_len < obs.shape[0]:
        valid_joint_data = obs[demo_len:]

        joint_data = joint_data[:demo_len]

    rms = RunningMeanStd(shape=joint_data.shape[1])
    rms.mean = obs_mean
    rms.var = obs_var
    env.obs_rms = rms
    print(rms.mean, rms.var)

    train_db = RND_Dataset(joint_data, normalizer=env)
    train_loader = DataLoader(train_db, batch_size=opt.batch_size, shuffle=True, num_workers=4, drop_last=False)

    if valid_joint_data is not None:
        valid_db = RND_Dataset(valid_joint_data, normalizer=env)
        valid_loader = DataLoader(valid_db, batch_size=opt.batch_size*2, shuffle=False, num_workers=4, drop_last=False)

    rnd = RND(joint_data.shape[1], env, pred_arch=[20, 20], target_arch=[20])

    rnd.train(train_loader, opt.epochs, 1e-3)

    res = []
    for data in train_loader:
        res.append(cuda_to_np(rnd.reward(data.cuda())))

    res = np.concatenate(res)
    print(res)


    obs_rms = RunningMeanStd(shape=env.observation_space.shape)
    obs_rms.mean = np.mean(obs, axis=0)
    obs_rms.var = np.var(obs, axis=0)
    print(obs_rms.mean)
    #stablize the rms hack
    obs_rms.count = 100000

    opt.epochs = 200
    opt.n_trajs = 4
    opt.lr = 3e-4
    opt.batch_size = 128
    # model = None
    model = bc_from_demo_v2(opt)


    return rnd, env, BC_Dataset((obs, actions)), obs_rms, model


@hydra.main(config_path="../config", config_name="collect_demo.yaml")
def bc_demo_hopper(opt: DictConfig):
    bc_from_demo_v2(opt)


@hydra.main(config_path="../config", config_name="bc_reacher.yaml")
def bc_demo_reacher(opt: DictConfig):
    bc_from_demo_v2(opt)


@hydra.main(config_path="../config", config_name="bc_reacher.yaml")
def rnd_reward(opt: DictConfig):
    rnd_from_demo(opt)


def get_train_data(opt):
    with open(f"/home/ruohan/projects/baselines/data/{opt.env_id[:-3]}-v2.pkl", "rb") as f:
        state_dict = pickle.load(f)

    traj_length = opt.traj_length
    obs, actions = state_dict["observations"], np.squeeze(state_dict["actions"])

    valid_obs = None

    env = VecNormalize(make_vec_env(opt.env_id), training=False)
    print("Actions clipped")
    actions = np.clip(actions, env.action_space.low, env.action_space.high)

    obs_mean = np.mean(obs, axis=0)
    obs_var = np.var(obs, axis=0)

    demo_len = opt.n_trajs * traj_length
    if demo_len < obs.shape[0]:
        valid_obs = obs[demo_len:]
        valid_actions = actions[demo_len:]

        obs = obs[:demo_len]
        actions = actions[:demo_len]

    rms = RunningMeanStd(shape=env.observation_space.shape)
    rms.mean = obs_mean
    rms.var = obs_var
    rms.count = 100000
    env.obs_rms = rms

    return BC_Dataset((obs, actions), normalizer=env)


if __name__ == '__main__':
    # collect_demo()
    # bc_from_demo()
    # bc_sanity_check()

    # bc_from_demo_v2()

    # bc_demo_hopper()

    rnd_reward()

    # bc_demo_reacher()

