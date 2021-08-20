import torch
from torch import nn, optim
import numpy as np

from util import np_to_cuda

class RandomWD(nn.Module):
    def __init__(self, feat_dim, obs_space, act_space, sigma=1., gamma=1., lr=1e-3):
        super(RandomWD, self).__init__()
        self.feat_dim = feat_dim
        self.rf_W = torch.randn((obs_space.shape[0] + act_space.shape[0], feat_dim), requires_grad=False, device="cuda")/sigma
        self.rf_b = torch.rand((1, feat_dim), requires_grad=False, device="cuda") * 2 * np.pi
        self.rf_scale = np.sqrt(2./ self.feat_dim)

        self.beta_1 = nn.Linear(feat_dim, 1, bias=False)
        self.beta_2 = nn.Linear(feat_dim, 1, bias=False)

        self.gamma = gamma

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def get_random_feature(self, obs, actions): #x is of shape batch_size x
        x = torch.cat([obs, actions], dim=1)
        return torch.cos(torch.matmul(x, self.rf_W) + self.rf_b) * self.rf_scale, x

    def wd(self, obs_x, actions_x, obs_y, actions_y):
        x_feat, x_cat = self.get_random_feature(obs_x, actions_x)
        y_feat, y_cat = self.get_random_feature(obs_y, actions_y)

        x_score = self.beta_1(x_feat)
        y_score = self.beta_2(y_feat)
        l2_dist = torch.square(x_cat - y_cat).sum(dim=1, keepdim=True)
        dist = x_score - y_score + self.gamma * torch.exp((x_score - y_score - l2_dist)/ self.gamma)

        return torch.mean(dist)

    def update(self, obs_x, actions_x, obs_y, actions_y):
        dist = - self.wd(obs_x, actions_x, obs_y, actions_y) # maximize the dist when updating beta parameters
        print(dist)

        self.optimizer.zero_grad()
        dist.backward()
        self.optimizer.step()


class WDMetric(object):
    def __init__(self, train_data, env, wd_model):
        self.obs, self.actions = train_data
        self.wd_model = wd_model
        self.env = env

        self.size = self.obs.shape[0]

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(self.size), batch_size)
        obs = self.obs[idx]
        actions = self.actions[idx]

        if self.env:
            obs = self.env.normalize_obs(obs)

        return np_to_cuda(obs), np_to_cuda(actions)

    def __call__(self, obs_x, actions_x):
        obs_y, actions_y = self.sample(obs_x.shape[0])

        return self.wd_model.wd(obs_x, actions_x, obs_y, actions_y)

    def update_wd(self, obs_x, actions_x):
        obs_y, actions_y = self.sample(obs_x.shape[0])

        self.wd_model.update(obs_x, actions_x, obs_y, actions_y)








