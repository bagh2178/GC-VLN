import argparse
import random
import numpy as np
import torch

from src.agent import Agent, init_env
from habitat.config.default import get_config
from config.default import _C as POLICY_CONIFG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument('--local_rank', type=int, default=0, help="local gpu id")
    parser.add_argument('--split_num', type=int, default=1, help="dataset episode split number")
    parser.add_argument('--split_index', type=int, default=0, help="dataset episode split index")
    parser.add_argument('--GSAM2_server_port', type=int, default=7003, help="GSAM2 server port")
    parser.add_argument('--dataset', type=str, choices=['r2r', 'rxr'], default='rxr', help="Dataset choice: r2r or rxr")
    parser.add_argument('--experiment_id', type=str, default='default_experiment', help="Experiment ID for logging")
    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, opts=None, local_rank=None, split_num=1, split_index=0, GSAM2_server_port=7003, dataset='rxr', experiment_id='default_experiment') -> None:
    r"""Runs experiment given config_path and options

    Args:
        exp_config: path to config file.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts, POLICY_CONIFG)

    config.defrost()
    config.local_rank = local_rank
    config.DATASET.SPLIT_NUM = split_num
    config.DATASET.SPLIT_INDEX = split_index
    config.DATASET.DATASET_TYPE = dataset
    config.DATASET.EXPERIMENT_ID = experiment_id
    config.POLICY_CONFIG.GSAM2_SERVER_PORT = GSAM2_server_port
    config.freeze()

    # this parameter is in the get_config function
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    # initialize environment and agent
    env = init_env(config)
    agent = Agent(env, config)
    agent.act()

if __name__ == "__main__":
    main()
