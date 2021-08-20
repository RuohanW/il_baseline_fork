import numpy as np
import torch
import random
import pickle

from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3.common.env_util import make_vec_env

class ExpertDataset:
    def __init__(self, env_name, num_trajs, seed, ensemble_shuffle_type):
        self.env_name = env_name
        self.num_trajs = num_trajs
        self.seed = seed
        self.ensemble_shuffle_type = ensemble_shuffle_type


    def load_demo_data(self, training_data_split, batch_size, ensemble_size):
        with open(f"/home/ruohan/projects/baselines/data/{self.env_name[:-3]}-v2.pkl", "rb") as f:
            state_dict = pickle.load(f)

        obs, acs = state_dict["observations"], np.squeeze(state_dict["actions"])

        env = make_vec_env(self.env_name)
        print("Actions clipped")
        acs = np.clip(acs, env.action_space.low, env.action_space.high)

        perm = np.random.permutation(obs.shape[0])
        obs = obs[perm]
        acs = acs[perm]

        n_train = int(obs.shape[0]*training_data_split)
        obs_train = obs[:n_train]
        acs_train = acs[:n_train]
        obs_test  = obs[n_train:]
        acs_test  = acs[n_train:]

        if self.ensemble_shuffle_type == 'norm_shuffle' or ensemble_size is None:
            shuffle = True
        elif self.ensemble_shuffle_type == 'no_shuffle' and ensemble_size is not None:
            shuffle = False
        elif self.ensemble_shuffle_type == 'sample_w_replace' and ensemble_size is not None:
            print('***** sample_w_replace *****')
            # sample with replacement
            obs_train_resamp, acs_train_resamp = [], []
            for k in range(n_train * ensemble_size):
                indx = random.randint(0, n_train - 1)
                obs_train_resamp.append(obs_train[indx])
                acs_train_resamp.append(acs_train[indx])
            obs_train = np.stack(obs_train_resamp)
            acs_train = np.stack(acs_train_resamp)
            shuffle = False

        tr_batch_size = min(batch_size, len(obs_train))
        # If Droplast is False, insure that that dataset is divisible by
        # the number of polices in the ensemble
        tr_drop_last = (tr_batch_size!=len(obs_train))
        if not tr_drop_last and ensemble_size is not None:
            tr_batch_size = int(ensemble_size * np.floor(tr_batch_size/ensemble_size))
            obs_train = obs_train[:tr_batch_size]
            acs_train = acs_train[:tr_batch_size]

        obs_train = torch.from_numpy(obs_train)
        acs_train = torch.from_numpy(acs_train)
        trdata = DataLoader(TensorDataset(obs_train, acs_train),\
                           batch_size = tr_batch_size, shuffle=shuffle, drop_last=tr_drop_last)

        if len(obs_test) == 0:
            tedata = None
        else:
            te_batch_size = min(batch_size, len(obs_test))
            # If Droplast is False, insure that that dataset is divisible by
            # the number of polices in the ensemble
            te_drop_last = (te_batch_size!=len(obs_test))
            if not te_drop_last and ensemble_size is not None:
                te_batch_size = int(ensemble_size * np.floor(te_batch_size/ensemble_size))
                obs_test = obs_test[:te_batch_size]
                acs_test = acs_test[:te_batch_size]

            obs_test = torch.from_numpy(obs_test)
            acs_test = torch.from_numpy(acs_test)
            tedata = DataLoader(TensorDataset(obs_test, acs_test),\
                                batch_size = te_batch_size, shuffle=shuffle, drop_last=te_drop_last)
        return {'trdata':trdata, 'tedata': tedata}

