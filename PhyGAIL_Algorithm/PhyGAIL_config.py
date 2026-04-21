import os
import sys
import time
import shutil
import warnings
import faulthandler
faulthandler.enable()

from Configurations import *

warnings.filterwarnings('ignore')
current_time = time.localtime()

# ==============================
# Configuration Parameters
# ==============================
node_feat_dim = 2*config_dimension+1 # position + velocity + degree
hidden_dim = 128
num_gnn_layers = 3
dropout = 0.1

# PPO parameters
ppo_clip = 0.2
ppo_epochs = 5
entropy_coef = 0.05
min_entropy_coef = 0.005
value_coef = 0.5
gae_lambda = 0.95
gamma = 0.99
log_std_min = -2
log_std_max = 0.5
actor_max_norm = 1.0
critic_max_norm = 1.0

# GAIL parameters
gail_coef = 0.1

# Reward setting
base_success_reward = 20
base_time_reward = 20
base_connect_reward = 2.5
base_subnet_penalty = 5.0

# Training parameters
batch_size = 512
target_batch_size = 4096         # Target batch size for PPO updates
actor_lr = 1e-4
critic_lr = 1e-4
discriminator_lr = 5e-5
discriminator_epoch = 1
weight_decay = 5e-5
discriminator_weight_decay = 5e-5
horizon = int(max(1000.0, config_width))  # Steps per rollout

# Number of training epochs
num_epochs = 1000
num_envs = 32  # Number of parallel environments
checkpoint_epoch = 200  # Checkpoint save interval
visualize_epoch = 50  # Visualization interval
eval_interval = 5  # Evaluation interval

# Multi-GPU settings
device_ids = [0, 1]
device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')
if hasattr(args, 'cuda_device') and args.cuda_device is not None:
    device = torch.device(f'cuda:{device_ids[args.cuda_device]}' if torch.cuda.is_available() else 'cpu')

print(f"Using CUDA device: {device}")

# Parse command line arguments
run_debug = args.debug
resume_from_epoch = args.resume
is_training = args.training
is_overwrite = args.overwrite

# Path settings for data, trajectories, and model saving
# config_num_of_agents should be defined in Configurations.py
# args should be parsed from command line arguments in the main script
data_path = f"Database/expert_solutions_{config_num_of_agents}_train.npz"
trajectory_path = f"Database/expert_trajectories_{config_num_of_agents}_train.npz"

save_dir = f"artifacts/swarm_{config_num_of_agents}/{current_time.tm_mon}.{current_time.tm_mday}"
if hasattr(args, 'log') and args.log is not None:
    save_dir = f"artifacts/swarm_{config_num_of_agents}/{current_time.tm_mon}.{current_time.tm_mday}_{args.log}"
if os.path.exists(save_dir) and resume_from_epoch is None:
    if is_overwrite:
        print(f"Warning: Save directory already exists. Overwrite: {save_dir}.")
        shutil.rmtree(save_dir)
    else:
        count = 1
        new_save_dir = f"{save_dir}_{count}"
        while os.path.exists(new_save_dir):
            count += 1
            new_save_dir = f"{save_dir}_{count}"
        save_dir = new_save_dir
        print(f"Warning: Save directory already exists. Create new directory: {save_dir}.")

log_dir = f"{save_dir}/train.log"
fig_save_dir = f"{save_dir}/visualizations"
model_save_path = f"{save_dir}/best_model_{config_num_of_agents}.pth"

# pretrained model path
use_pretrained = False
pretrained_path = "models/model_best.pth"

# Debug mode: use smaller settings for quick testing
if run_debug:
    data_path = f"Database/expert_solutions_20_debug.npz"
    trajectory_path = f"Database/expert_trajectories_20_debug.npz"
    save_dir = f"artifacts/debug"
    log_dir = f"{save_dir}/debug.log"
    fig_save_dir = f"{save_dir}/visualizations"
    model_save_path = f"{save_dir}/best_model.pth"
    batch_size = 64
    target_batch_size = 128
    num_epochs = 2
    ppo_epochs = 2
    discriminator_epoch = 1
    num_envs = 2
    horizon = 100
    visualize_epoch = 1
    eval_interval = 1

# Resume training: adjust log path to avoid overwriting existing logs
if resume_from_epoch is not None:
    log_dir = f"{save_dir}/train_resume_{resume_from_epoch}.log"

# Create save directories
if is_training:
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(fig_save_dir, exist_ok=True)
    sys.stdout = open(log_dir, 'w')
