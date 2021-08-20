import sys

from dril_pack.expert_dataset import ExpertDataset
from dril_pack.ensemble import Ensemble

from stable_baselines3.common.env_util import make_vec_env
from dril_pack.dril import DRIL
import hydra

@hydra.main(config_path="../config", config_name="dril_test.yaml")
def get_dril_reward(args):
    expert_dataset = ExpertDataset(args.env_name,
                                   args.num_trajs, args.seed, args.ensemble_shuffle_type)

    # Train or load ensemble policy
    envs = make_vec_env(args.env_name)
    ensemble_policy = Ensemble(device=args.device, envs=envs,
                               expert_dataset=expert_dataset,
                               uncertainty_reward=args.dril_uncertainty_reward,
                               ensemble_hidden_size=args.ensemble_hidden_size,
                               ensemble_drop_rate=args.ensemble_drop_rate,
                               ensemble_size=args.ensemble_size,
                               ensemble_batch_size=args.ensemble_batch_size,
                               ensemble_lr=args.ensemble_lr,
                               num_ensemble_train_epoch=args.num_ensemble_train_epoch,
                               num_trajs=args.num_trajs,
                               seed=args.seed,
                               env_name=args.env_name,
                               training_data_split=args.training_data_split,
                               save_model_dir=args.save_model_dir,
                               save_results_dir=args.save_results_dir)

    dril = DRIL(device=args.device, envs=envs, ensemble_policy=ensemble_policy,
                dril_bc_model=None, expert_dataset=expert_dataset,
                ensemble_quantile_threshold=args.ensemble_quantile_threshold,
                ensemble_size=args.ensemble_size, dril_cost_clip=args.dril_cost_clip,
                env_name=args.env_name, num_dril_bc_train_epoch=1,
                training_data_split=args.training_data_split)

    return dril, envs


if __name__ == '__main__':
    get_dril_reward()