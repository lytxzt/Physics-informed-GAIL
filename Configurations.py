import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--training', action='store_true')
parser.add_argument('--resume', type=int, default=None, help='Path to checkpoint to resume from')
parser.add_argument('--num_agents', type=int, help='Override number of agents')
parser.add_argument('--dimension', type=int, help='Override dimension')
parser.add_argument('--cuda_device', type=int, help='Override cuda device')
parser.add_argument('--log', type=str, help='Override experiment count')
parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing save directory if it exists')
args, _unknown_args = parser.parse_known_args()

# Respect an existing CUDA setting from the shell, otherwise expose the
# first two devices by default.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

seed = 3407
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

config_num_of_agents = 20
config_communication_range = 120.0
config_dimension = 2


if hasattr(args, 'debug') and args.debug:
    config_num_of_agents = 20
    config_dimension = 2
    print(f"[Debug] Using debug num_of_agents: {config_num_of_agents}, dimension: {config_dimension}")
else:
    if hasattr(args, 'num_agents') and args.num_agents is not None:
        config_num_of_agents = int(args.num_agents)
        print(f"[Config] Reset num_of_agents to {config_num_of_agents}")
    if hasattr(args, 'dimension') and args.dimension is not None:
        config_dimension = int(args.dimension)
        print(f"[Config] Reset dimension to {config_dimension}")

config_initial_swarm_positions = pd.read_excel(f"Configurations/swarm_positions_{config_num_of_agents}.xlsx")
config_initial_swarm_positions = config_initial_swarm_positions.values[:, 1:1+config_dimension]
config_initial_swarm_positions = np.array(config_initial_swarm_positions, dtype=np.float64)

config_range_list = {"10":220, "20":320.0, "50":500.0, "100":750.0, "200":1000.0, "500":1600.0, "1000":2250.0}
config_width = config_range_list[f'{config_num_of_agents}']
config_length = config_range_list[f'{config_num_of_agents}']
config_height = 100.0

config_maximum_step = int(0.8 * config_width)

config_space_range = np.array([config_width, config_length]) if config_dimension == 2 else np.array([config_width, config_length, config_height])
config_central_point = config_space_range * 0.5

config_max_speed = 10.0
config_dt = 0.1
config_constant_speed = config_max_speed * config_dt

config_maximum_destroy_num = 50
config_minimum_remain_num = 2

config_meta_training_epi = 500
config_K = 1 / 100
config_best_eta = 0.3
config_best_epsilon = 0.99

config_num_destructed_UAVs = 50
config_normalize_positions = True

config_alpha_k = [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 0.9, 0.95, 1, 1.5, 2, 3, 5]
config_gcn_repeat = 100
config_expension_alpha = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
config_d0_alpha = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

config_representation_step = 450

config_random_seed = range(0, 1000)

