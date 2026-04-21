import os
import numpy as np
import torch
import torch.optim as optim
import torch.multiprocessing as mp

from Configurations import *
from PhyGAIL_Algorithm.PhyGAIL_config import *
from PhyGAIL_Algorithm.PhyGAIL_utils import generate_expert_trajectories
from PhyGAIL_Algorithm.PhyGAIL_dataloader import ExpertDataManager, SubnetGraphBuilder
from PhyGAIL_Algorithm.PhyGAIL_network import ActorNetwork, DiscriminatorNetwork, CriticNetwork
from PhyGAIL_Algorithm.PhyGAIL_framework import PPOPhyGAILTrainer
from PhyGAIL_Algorithm.PhyGAIL_env import SwarmEnv, ParallelEnv

def main_parallel():
    """Train PhyGAIL with parallel environments."""
    print("=" * 60)
    print("PPO & PhyGAIL for UAV Swarm Recovery")
    print("Framework: Actor-Critic + Discriminator + Parallel Environments")
    print("Dataloader: Global index + Lazy loading")
    print("Network: Physics Gated GNN + MLP")
    print("Running Envs: Parallel on single GPU")
    print("=" * 60)
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    if torch.cuda.device_count() >= 2:
        print(f"Using GPUs: {device_ids}")
    else:
        print("Warning: Using single GPU.")
    
    try:
        mp.set_start_method('fork', force=True)
        print("Set multiprocessing start method to 'fork'")
    except RuntimeError:
        print("Multiprocessing context already set")
    
    if not os.path.exists(data_path):
        print(f"Error: Expert data not found at {data_path}")
        print("Please run Collect_Expert_Data.py first to generate the dataset.")
        return
    
    if not os.path.exists(trajectory_path):
        print("\n1. Generating expert trajectories...")
        trajectories = generate_expert_trajectories(max_speed=config_max_speed, dt=config_dt)
    else:
        print("\n1. Loading expert trajectory data...")
        loaded = np.load(trajectory_path, allow_pickle=True)
        if 'data' in loaded:
            trajectories = loaded['data'].tolist()
        else:
            trajectories = loaded['arr_0'].tolist()
        print(f"Loaded {len(trajectories)} expert trajectories")
    
    graph_builder = SubnetGraphBuilder()
    
    print("\n2. Creating expert data manager...")
    expert_data_manager = ExpertDataManager(trajectories, graph_builder, augmentation_factor=8)
    print(f"Created expert data manager with {len(expert_data_manager.augmented_trajectories)} total transitions")
    
    print("\n3. Creating environment constructors...")
    env_fns = []
    for i in range(num_envs):
        def make_env(manager=expert_data_manager):
            return SwarmEnv(graph_builder, manager)
        env_fns.append(make_env)
    
    print(f"Created {len(env_fns)} environment constructors")
    
    print("\n4. Creating parallel environments...")
    parallel_env = ParallelEnv(env_fns, device='cpu', base_seed=3407)
    print(f"Created parallel environment with {parallel_env.num_envs} workers")
    
    print("\n5. Creating networks...")
    actor = ActorNetwork(action_dim=config_dimension, hidden_dim=hidden_dim)
    critic = CriticNetwork(hidden_dim=hidden_dim)
    discriminator = DiscriminatorNetwork(action_dim=config_dimension, hidden_dim=hidden_dim)
    
    print(f"Actor parameters: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    print("\n6. Creating optimized PPO+PhyGAIL trainer...")
    trainer = PPOPhyGAILTrainer(actor, critic, discriminator, parallel_env, expert_data_manager, device)

    start_epoch = 0
    if resume_from_epoch is not None:
        checkpoint_path = f"{save_dir}/checkpoint_epoch_{resume_from_epoch}.pth"
        print(f"\n>>> Resuming training from checkpoint: {checkpoint_path} ...")
        start_epoch = trainer.load_model(checkpoint_path)
    elif use_pretrained:
        if os.path.exists(pretrained_path):
            print(f"\n>>> Loading pretrained model from {pretrained_path} for Curriculum Learning...")
            trainer.load_model(pretrained_path)
            print(">>> Resetting optimizer states for new training phase...")
            trainer.actor_optimizer = optim.AdamW(trainer.actor.parameters(), lr=actor_lr, weight_decay=weight_decay)
            trainer.critic_optimizer = optim.AdamW(trainer.critic.parameters(), lr=critic_lr, weight_decay=weight_decay)
            trainer.discriminator_optimizer = optim.AdamW(trainer.discriminator.parameters(), lr=discriminator_lr, weight_decay=discriminator_weight_decay)
            trainer.reset_training_state()
    
    print("\n7. Starting training with parallel environments...")
    trainer.train(num_epochs, start_epoch=start_epoch)
    
    print("\n" + "=" * 60)
    print("Optimized PPO+PhyGAIL training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main_parallel()
