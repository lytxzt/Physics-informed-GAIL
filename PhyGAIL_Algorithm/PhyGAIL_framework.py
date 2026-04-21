import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import cdist
from copy import deepcopy
from tqdm import tqdm

from Configurations import *
from PhyGAIL_Algorithm.PhyGAIL_config import *
from PhyGAIL_Algorithm.PhyGAIL_utils import fast_batch_from_dicts, fast_to_data_list, plot_training_curves
from PhyGAIL_Algorithm.PhyGAIL_env import SwarmEnv

# ==============================
# Replay Buffer for PPO
# ==============================
class RolloutBuffer:
    """Temporary storage for one PPO rollout."""
    def __init__(self):
        self.buffer = []
        
    def push(self, experience):
        """Store a single-step experience tuple."""
        # experience: (state, action, log_prob, value, reward, done, next_state)
        self.buffer.append(experience)
    
    def get_all(self):
        """Return all rollout entries."""
        return self.buffer
    
    def clear(self):
        """Clear the buffer (must be called after PPO update)."""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)

# ==============================
# PPOPhyGAILTrainer (Batch Collection)
# ==============================
class PPOPhyGAILTrainer:
    """PhyGAIL trainer with parallel rollout collection and PPO updates."""

    def __init__(self, actor, critic, discriminator, envs, expert_data_manager, device):
        self.device = device
        
        # Use parallel environment container
        self.envs = envs
        self.expert_data_manager = expert_data_manager
        self.steps_per_epoch = horizon
        
        # Initialize models
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.discriminator = discriminator.to(device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(),
            lr=actor_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(),
            lr=critic_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        self.discriminator_optimizer = optim.AdamW(
            self.discriminator.parameters(),
            lr=discriminator_lr,
            weight_decay=discriminator_weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Initialize replay buffer
        self.replay_buffer = RolloutBuffer()
        
        # Record training history
        self.actor_losses = []
        self.critic_losses = []
        self.discriminator_losses = []
        self.episode_rewards = []
        self.success_rates = []
        self.current_ep_rewards = np.zeros(envs.num_envs, dtype=np.float32)

        # Initialize best metrics
        self.best_success_rate = -1.0
        self.best_avg_reward = -float('inf')

        self.reward_components = ['env_total', 'gail', 'step', 'collision', 'terminal']
        self.ep_reward_breakdown_history = {k: [] for k in self.reward_components}
        self.current_ep_reward_breakdown = {k: np.zeros(envs.num_envs, dtype=np.float32) for k in self.reward_components}

        self.history = self._build_empty_history()

    @staticmethod
    def _build_empty_history():
        """Return an empty training-history dictionary with the expected keys."""
        return {
            'epoch': [],
            'actor_loss': [],
            'critic_loss': [],
            'discriminator_loss': [],
            'reward': [],
            'env_reward': [],
            'gail_reward': [],
            'success_rate': [],
            'val_epoch': [],
            'val_reward': [],
            'val_success_rate': []
        }

    def reset_training_state(self):
        """Reset best metrics and history when starting a new training phase."""
        self.best_success_rate = -1.0
        self.best_avg_reward = -float('inf')
        self.history = self._build_empty_history()
        
    def collect_trajectories_batch(self, epoch, num_epochs):
        """Collect one rollout window from all parallel environments."""
        
        states_list = []
        rewards_list = []
        dones_list = []
        values_list = []
        node_log_probs_list = [] 
        actions_list_store = []
        node_counts_list = [] 
        active_batch_list = []
        active_agent_indices_list = []
        
        raw_state_dicts = self.envs.reset()
        state = fast_batch_from_dicts(raw_state_dicts, self.device)
        if state is None: return [] 
        
        iterator = tqdm(range(self.steps_per_epoch), desc="Collect", leave=False)
        
        for _ in iterator:
            if hasattr(state, 'remain_mask'):
                active_batch = state.batch[state.remain_mask]
                active_agent_indices = state.subnet_indices
            else:
                active_batch = state.batch
                active_agent_indices = None

            current_node_counts = torch.bincount(active_batch, minlength=self.envs.num_envs).cpu()
            node_counts_list.append(current_node_counts)

            with torch.no_grad():
                actions, node_log_probs = self.actor(state)
                value = self.critic(state) 
                disc_prob = self.discriminator(state, actions)
                node_gail_reward = -torch.log(torch.clamp(1 - disc_prob, min=1e-8, max=0.99)).squeeze()

            action_np = actions.cpu().numpy()
            actions_list_np = []
            cursor = 0
            for count in current_node_counts:
                c = count.item()
                actions_list_np.append(action_np[cursor : cursor + c].copy())
                cursor += c

            next_raw_dicts, env_rewards_list, dones, infos = self.envs.step(actions_list_np)

            if active_agent_indices is not None:
                rewards_matrix = np.array(env_rewards_list)
                batch_idx_np = active_batch.cpu().numpy()
                agent_idx_np = active_agent_indices.cpu().numpy()
                gathered_rewards = rewards_matrix[batch_idx_np, agent_idx_np]
                node_env_reward = torch.from_numpy(gathered_rewards).float().to(self.device)
            else:
                if isinstance(env_rewards_list[0], (np.ndarray, list)):
                    env_rewards_concat = np.concatenate(env_rewards_list)
                    node_env_reward = torch.from_numpy(env_rewards_concat).float().to(self.device)
                else:
                    env_rewards_tensor = torch.tensor(env_rewards_list, dtype=torch.float32, device=self.device)
                    node_env_reward = env_rewards_tensor[active_batch]

            scaled_env_reward = node_env_reward
            weighted_gail_reward = gail_coef * node_gail_reward
            raw_total_reward = scaled_env_reward + weighted_gail_reward
            static_scaled_reward = raw_total_reward / 20.0
            
            total_rewards = torch.clamp(static_scaled_reward, -5.0, 5.0)

            dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)
            
            states_list.append(state.cpu())
            actions_list_store.append(actions.cpu())
            node_log_probs_list.append(node_log_probs.cpu())
            rewards_list.append(total_rewards.cpu())
            values_list.append(value.squeeze().cpu())
            dones_list.append(dones_tensor.cpu())
            active_batch_list.append(active_batch.cpu())
            active_agent_indices_list.append(active_agent_indices.cpu())
            
            # Track per-environment reward means over alive nodes only.
            if active_agent_indices is not None and 'gathered_rewards' in locals():
                step_env_means = []
                step_gail_means = []
                ptr = 0
                gail_np = weighted_gail_reward.cpu().numpy()
                
                for count in current_node_counts:
                    c = count.item()
                    if c > 0:
                        env_active_rewards = gathered_rewards[ptr : ptr + c]
                        step_env_means.append(np.mean(env_active_rewards))
                        step_gail_means.append(np.mean(gail_np[ptr : ptr + c]))
                    else:
                        step_env_means.append(0.0)
                        step_gail_means.append(0.0)
                    ptr += c
                    
                self.current_ep_rewards += (np.array(step_env_means) + np.array(step_gail_means))
            else:
                self.current_ep_rewards += np.array([np.mean(r) if isinstance(r, (np.ndarray, list)) else r for r in env_rewards_list])

            for k in self.reward_components:
                if k == 'gail':
                    self.current_ep_reward_breakdown[k] += np.array(step_gail_means)
                elif k == 'env_total':
                    self.current_ep_reward_breakdown[k] += np.array(step_env_means)
                else:
                    comp_vals = np.array([info.get('reward_breakdown', {}).get(k, 0.0) for info in infos])
                    self.current_ep_reward_breakdown[k] += comp_vals

            for i, d in enumerate(dones):
                if d:
                    self.episode_rewards.append(self.current_ep_rewards[i])
                    self.current_ep_rewards[i] = 0
                    
                    for k in self.reward_components:
                        self.ep_reward_breakdown_history[k].append(self.current_ep_reward_breakdown[k][i])
                        self.current_ep_reward_breakdown[k][i] = 0.0
                    
                    info_src = infos[i].get('final_stat', infos[i])
                    is_success = info_src.get('connected', False) and (info_src.get('num_subnets', 1) == 1)
                    self.success_rates.append(1.0 if is_success else 0.0)

            state = fast_batch_from_dicts(next_raw_dicts, self.device)
            
        # Align node-wise values across steps before computing GAE.
        with torch.no_grad():
            next_value = self.critic(state).squeeze().cpu()
            if hasattr(state, 'remain_mask'):
                last_active_batch = state.batch[state.remain_mask]
                last_agent_indices = state.subnet_indices
            else:
                last_active_batch = state.batch
                last_agent_indices = None
            
        values_list.append(next_value)
        active_batch_list.append(last_active_batch.cpu())
        active_agent_indices_list.append(last_agent_indices.cpu())
        
        final_returns = [None] * len(rewards_list)
        final_advantages = [None] * len(rewards_list)
        
        num_envs = self.envs.num_envs
        max_agents = config_num_of_agents
        prev_gae_matrix = torch.zeros((num_envs, max_agents), dtype=torch.float32)
        next_v_matrix = torch.zeros((num_envs, max_agents), dtype=torch.float32)
        
        for step in reversed(range(len(rewards_list))):
            curr_r = rewards_list[step]
            curr_v = values_list[step]
            curr_dones = dones_list[step]
            
            next_batch_idx = active_batch_list[step + 1]
            next_agent_idx = active_agent_indices_list[step + 1]
            next_v_raw = values_list[step + 1]
            
            next_v_matrix.fill_(0)
            done_mask = curr_dones.numpy() > 0.5
            valid_next_mask = ~done_mask[next_batch_idx.numpy()]
            
            if valid_next_mask.any():
                v_b = next_batch_idx[valid_next_mask]
                v_a = next_agent_idx[valid_next_mask]
                next_v_matrix[v_b, v_a] = next_v_raw[valid_next_mask]
                
            curr_batch_idx = active_batch_list[step]
            curr_agent_idx = active_agent_indices_list[step]
            
            aligned_next_v = next_v_matrix[curr_batch_idx, curr_agent_idx]
            aligned_prev_gae = prev_gae_matrix[curr_batch_idx, curr_agent_idx]
            
            # Stop GAE propagation across episode boundaries.
            valid_gae_mask = ~done_mask[curr_batch_idx.numpy()]
            valid_gae_tensor = torch.from_numpy(valid_gae_mask).float()
            aligned_prev_gae = aligned_prev_gae * valid_gae_tensor
            
            delta = curr_r + gamma * aligned_next_v - curr_v
            curr_gae = delta + gamma * gae_lambda * aligned_prev_gae
            curr_ret = curr_gae + curr_v
            
            final_returns[step] = curr_ret
            final_advantages[step] = curr_gae
            
            prev_gae_matrix.fill_(0)
            prev_gae_matrix[curr_batch_idx, curr_agent_idx] = curr_gae
            
        values_list.pop()
        
        # Prepare final data list
        all_data_list = []
        for step in range(len(states_list)):
            batch_obj = states_list[step]
            graphs = fast_to_data_list(batch_obj)
            
            step_actions = actions_list_store[step]
            step_node_log_probs = node_log_probs_list[step]
            step_returns = final_returns[step]
            step_adv = final_advantages[step]
            step_vals = values_list[step]
            counts = node_counts_list[step]
            
            current_cursor = 0
            for env_i, graph in enumerate(graphs):
                n_alive = counts[env_i].item()
                
                graph.actions = step_actions[current_cursor : current_cursor + n_alive]
                if step_node_log_probs is not None:
                     graph.old_log_prob = step_node_log_probs[current_cursor : current_cursor + n_alive]
                
                graph.returns = step_returns[current_cursor : current_cursor + n_alive]
                graph.advantages = step_adv[current_cursor : current_cursor + n_alive]
                graph.value = step_vals[current_cursor : current_cursor + n_alive]
                
                current_cursor += n_alive
                all_data_list.append(graph)

        # Normalize advantages across the full epoch before minibatching.
        if len(all_data_list) > 0:
            all_advs = torch.cat([graph.advantages for graph in all_data_list])
            global_adv_mean = all_advs.mean()
            global_adv_std = all_advs.std() + 1e-8
            
            for graph in all_data_list:
                graph.advantages = (graph.advantages - global_adv_mean) / global_adv_std
                
        return all_data_list
    
    def update_discriminator(self, data_list):
        """Update the discriminator with gradient accumulation."""
        if len(data_list) == 0: return 0.0

        total_loss = 0
        steps = 0

        accumulation_steps = target_batch_size // batch_size
        
        random.shuffle(data_list)
        self.discriminator_optimizer.zero_grad()

        for i in tqdm(range(0, len(data_list), batch_size), desc="Update Discriminator", leave=False):
            p_mini = data_list[i : i + batch_size]
            if len(p_mini) == 0: continue

            policy_batch = Batch.from_data_list(p_mini).to(self.device)
            expert_batch = self.expert_data_manager.sample_batch(len(p_mini)).to(self.device)
            if expert_batch is None: continue

            expert_scores = self.discriminator(expert_batch, expert_batch.expert_action)
            policy_scores = self.discriminator(policy_batch, policy_batch.actions)
            
            expert_labels = torch.full_like(expert_scores, 0.9)
            policy_labels = torch.full_like(policy_scores, 0.1)
            
            loss = (F.binary_cross_entropy(expert_scores, expert_labels) + \
                    F.binary_cross_entropy(policy_scores, policy_labels)) / accumulation_steps

            loss.backward()
            total_loss += loss.item() * accumulation_steps
            steps += 1

            current_step = (i // batch_size) + 1
            is_update_step = (current_step % accumulation_steps == 0) or ((i + batch_size) >= len(data_list))
            
            if is_update_step:
                self.discriminator_optimizer.step()
                self.discriminator_optimizer.zero_grad()
            
            del policy_batch, expert_batch
            
        return total_loss / max(steps, 1)
    
    def update_actor_critic(self, data_list, epoch, num_epochs):
        """Update actor and critic using pre-batched graph data."""
        if len(data_list) == 0:
            return 0.0, 0.0
            
        actor_losses = []
        critic_losses = []
        
        batched_data = []
        
        random.shuffle(data_list)
        
        for i in range(0, len(data_list), batch_size):
            minibatch = data_list[i : i + batch_size]
            if len(minibatch) == 0: continue
            
            batch = Batch.from_data_list(minibatch)
            batch = batch.to(self.device)
            batched_data.append(batch)
            
        accumulation_steps = target_batch_size // batch_size
        if accumulation_steps < 1: accumulation_steps = 1

        for _ in tqdm(range(ppo_epochs), desc="Update PPO", leave=False):
            random.shuffle(batched_data)
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            accum_iter = 0

            for i, batch in enumerate(batched_data):
                b_actions = batch.actions
                b_log_probs = batch.old_log_prob
                b_returns = batch.returns
                b_adv = batch.advantages
                b_values = batch.value

                new_values = self.critic(batch).squeeze()
                huber_delta = 5.0
                critic_loss = F.huber_loss(new_values, b_returns, delta=huber_delta) / accumulation_steps

                node_features = self.actor.actor_encoder(batch)

                # Get remain_features
                if hasattr(batch, 'remain_mask'):
                    remain_features = node_features[batch.remain_mask]
                else:
                    remain_features = node_features
                
                if len(remain_features) > 0:
                    mean = self.actor.mean_head(remain_features)
                    
                    x = self.actor.log_std_head(remain_features)
                    log_std = torch.clamp(x, self.actor.log_std_min, self.actor.log_std_max)
                    std = torch.exp(log_std)
                    
                    normal = torch.distributions.Normal(mean, std)
                    
                    norm_actions = b_actions / config_max_speed
                    eps = 1e-4
                    norm_actions = torch.clamp(norm_actions, -1.0 + eps, 1.0 - eps)
                    raw_actions = torch.atanh(norm_actions)
                    log_prob_gauss = normal.log_prob(raw_actions)
                    correction = torch.log(1 - norm_actions.pow(2) + 1e-4).sum(dim=-1)
                    new_node_log_probs = log_prob_gauss.sum(dim=-1) - correction
                    node_entropy = normal.entropy().sum(dim=-1)
                    
                    ratios = torch.exp(new_node_log_probs - b_log_probs)
                    
                    surr1 = ratios * b_adv
                    surr2 = torch.clamp(ratios, 1 - ppo_clip, 1 + ppo_clip) * b_adv
                    
                    current_entropy_coef = max(min_entropy_coef, entropy_coef * (1 - epoch / (num_epochs * 0.5)))
                    actor_loss = (-torch.min(surr1, surr2).mean() - current_entropy_coef * node_entropy.mean()) / accumulation_steps
                else:
                    actor_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                critic_loss.backward()
                actor_loss.backward()
                
                actor_losses.append(actor_loss.item() * accumulation_steps)
                critic_losses.append(critic_loss.item() * accumulation_steps)
                
                accum_iter += 1
                
                is_update_step = (accum_iter % accumulation_steps == 0) or (i == len(batched_data) - 1)
                
                if is_update_step:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=critic_max_norm)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=actor_max_norm)
                    
                    self.critic_optimizer.step()
                    self.actor_optimizer.step()
                    
                    self.critic_optimizer.zero_grad()
                    self.actor_optimizer.zero_grad()
                    accum_iter = 0
                
        return np.mean(actor_losses) if actor_losses else 0.0, np.mean(critic_losses) if critic_losses else 0.0
    
    def train(self, num_epochs, start_epoch=0):
        """Run the full PPO+PhyGAIL training loop."""
        print(f"Starting PPO+PhyGAIL training on {self.device}...")
        print(f"Number of parallel environments: {self.envs.num_envs}")
        print(f"Horizon: {horizon}")
        print(f"Batch size: {batch_size}")

        self.validate_visualize(0)

        if use_pretrained:
            # Run deterministic evaluation
            val_reward, val_success, val_breakdown = self.evaluate_policy()
            print(f" [Eval] Pre-trained Model - Reward: {val_reward:.1f} "
                  f"(Col: {val_breakdown['collision']:.1f}, Step: {val_breakdown['step']:.1f}, "
                  f"Term: {val_breakdown['terminal']:.1f}) | Success: {val_success:.2%}")

        for epoch in range(start_epoch, num_epochs):
            
            self.actor.eval()
            self.critic.eval()
            self.discriminator.eval()
            data_list = self.collect_trajectories_batch(epoch, num_epochs)

            self.actor.train()
            self.critic.train()
            self.discriminator.train()
            
            discriminator_loss = 0.0
            if len(data_list) > 0:
                discriminator_loss = self.update_discriminator(data_list)
            
            actor_loss, critic_loss = self.update_actor_critic(data_list, epoch, num_epochs)
            
            self.actor_losses.append(actor_loss)
            self.critic_losses.append(critic_loss)
            self.discriminator_losses.append(discriminator_loss)
            
            avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
            success_rate = np.mean(self.success_rates[-100:]) if len(self.success_rates) >= 100 else np.mean(self.success_rates)

            avg_breakdown = {}
            for k in self.reward_components:
                hist = self.ep_reward_breakdown_history[k]
                avg_breakdown[k] = np.mean(hist[-100:]) if len(hist) > 0 else 0.0

            self.history['epoch'].append(epoch)
            self.history['actor_loss'].append(actor_loss)
            self.history['critic_loss'].append(critic_loss)
            self.history['discriminator_loss'].append(discriminator_loss)
            self.history['reward'].append(avg_reward)
            self.history['env_reward'].append(avg_breakdown['env_total'])
            self.history['gail_reward'].append(avg_breakdown['gail'])
            self.history['success_rate'].append(success_rate)
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Actor: {actor_loss:.4f}, Critic: {critic_loss:.4f}, Disc: {discriminator_loss:.4f}| "
                  f"Rew: {avg_reward:.1f} (Env: {avg_breakdown['env_total']:.1f}, GAIL: {avg_breakdown['gail']:.1f} | "
                  f"Col: {avg_breakdown['collision']:.1f}, Step: {avg_breakdown['step']:.1f}, "
                  f"Term: {avg_breakdown['terminal']:.1f}) | "
                  f"Succ: {success_rate:.2%}")
                        
            save_triggered = False
            if (epoch + 1) % eval_interval == 0:
                
                val_reward, val_success, val_breakdown = self.evaluate_policy()
                
                print(f" [Eval] Deterministic - Reward: {val_reward:.1f} "
                      f"(Col: {val_breakdown['collision']:.1f}, Step: {val_breakdown['step']:.1f}, "
                      f"Term: {val_breakdown['terminal']:.1f}) | Success: {val_success:.2%}")
                
                min_save_threshold = 0.6
                
                if val_success > self.best_success_rate and val_success >= min_save_threshold:
                    self.best_success_rate = val_success
                    self.best_avg_reward = val_reward
                    save_triggered = True
                    print(f" >>> New Best Model (Eval)! Success: {val_success:.2%}")
                    
                elif abs(val_success - self.best_success_rate) < 1e-4 and val_reward > self.best_avg_reward and val_success >= min_save_threshold:
                    self.best_avg_reward = val_reward
                    save_triggered = True
                    print(f" >>> New Best Model (Eval)! Higher Reward: {val_reward:.2f}")
                
                if save_triggered:
                    self.save_model(model_save_path, epoch=epoch+1)
                
                self.history['val_epoch'].append(epoch)
                self.history['val_reward'].append(val_reward)
                self.history['val_success_rate'].append(val_success)

                np.savez(f'{save_dir}/training_history.npz', 
                         epoch=self.history['epoch'],
                         actor_loss=self.history['actor_loss'],
                         critic_loss=self.history['critic_loss'],
                         discriminator_loss=self.history['discriminator_loss'],
                         reward=self.history['reward'],
                         env_reward=self.history['env_reward'],
                         gail_reward=self.history['gail_reward'],
                         success_rate=self.history['success_rate'],
                         val_epoch=self.history['val_epoch'],
                         val_reward=self.history['val_reward'],
                         val_success_rate=self.history['val_success_rate'])

            if (epoch + 1) % visualize_epoch == 0 or save_triggered:
                self.validate_visualize(epoch + 1)
                window_size  = 50 if (epoch + 1) >= 50 else epoch + 1
                plot_training_curves(npz_file_path=f'{save_dir}/training_history.npz', save_path=f'{save_dir}/training_curves.png', window_size=window_size)
            
            if (epoch + 1) % checkpoint_epoch == 0:
                 ckpt_path = f"{save_dir}/checkpoint_epoch_{epoch+1}.pth"
                 self.save_model(ckpt_path, epoch=epoch+1)
            
            if epoch % 50 == 0:
                self.replay_buffer.clear()
        
        self.envs.close()

        print("Training completed!")

    def evaluate_policy(self, num_steps=None):
        """Evaluate the current policy with deterministic actions."""
        if num_steps is None:
            num_steps = self.steps_per_epoch 

        self.actor.eval()
        
        total_rewards = []
        success_counts = 0
        total_episodes = 0
        
        current_ep_rewards = np.zeros(self.envs.num_envs, dtype=np.float32)
        
        eval_reward_components = ['env_total', 'step', 'collision', 'terminal']
        current_ep_breakdown = {k: np.zeros(self.envs.num_envs, dtype=np.float32) for k in eval_reward_components}
        total_breakdown_history = {k: [] for k in eval_reward_components}
        
        raw_state_dicts = self.envs.reset()
        state = fast_batch_from_dicts(raw_state_dicts, self.device)
        
        iterator = tqdm(range(num_steps), desc="Evaluating (Deterministic)", leave=False)
        
        for _ in iterator:
            with torch.no_grad():
                actions, _ = self.actor(state, deterministic=True)
                
            if hasattr(state, 'remain_mask'):
                active_batch = state.batch[state.remain_mask]
            else:
                active_batch = state.batch
            
            node_counts = torch.bincount(active_batch, minlength=self.envs.num_envs).cpu()
            
            action_np = actions.cpu().numpy()
            actions_list_np = []
            cursor = 0
            for count in node_counts:
                c = count.item()
                actions_list_np.append(action_np[cursor : cursor + c].copy())
                cursor += c
                
            next_raw_dicts, env_rewards_list, dones, infos = self.envs.step(actions_list_np)
            
            step_rewards_list = []
            for i, r in enumerate(env_rewards_list):
                if isinstance(r, (np.ndarray, list)):
                    n_active = node_counts[i].item()
                    
                    if n_active > 0:
                        avg_r = np.sum(r) / n_active
                        step_rewards_list.append(avg_r)
                    else:
                        step_rewards_list.append(0.0)
                else:
                    step_rewards_list.append(r)
            
            step_rewards = np.array(step_rewards_list)
            current_ep_rewards += step_rewards

            for k in eval_reward_components:
                if k == 'env_total':
                    current_ep_breakdown[k] += step_rewards
                else:
                    comp_vals = np.array([info.get('reward_breakdown', {}).get(k, 0.0) for info in infos])
                    current_ep_breakdown[k] += comp_vals
            
            for i, d in enumerate(dones):
                if d:
                    total_rewards.append(current_ep_rewards[i])
                    current_ep_rewards[i] = 0
                    total_episodes += 1

                    for k in eval_reward_components:
                        total_breakdown_history[k].append(current_ep_breakdown[k][i])
                        current_ep_breakdown[k][i] = 0.0
                    
                    info_src = infos[i].get('final_stat', infos[i])
                    is_success = info_src.get('connected', False) and (info_src.get('num_subnets', 1) == 1)
                    if is_success:
                        success_counts += 1
            
            state = fast_batch_from_dicts(next_raw_dicts, self.device)
            
        if total_episodes > 0:
            val_avg_reward = np.mean(total_rewards)
            val_success_rate = success_counts / total_episodes
            val_avg_breakdown = {k: np.mean(total_breakdown_history[k]) for k in eval_reward_components}
        else:
            val_avg_reward = np.mean(current_ep_rewards)
            val_success_rate = 0.0
            val_avg_breakdown = {k: np.mean(current_ep_breakdown[k]) for k in eval_reward_components}
            
        return val_avg_reward, val_success_rate, val_avg_breakdown
    
    def save_model(self, path, epoch=0):
        """Save model, including training progress and best metrics"""
        torch.save({
            'epoch': epoch,
            'best_success_rate': float(self.best_success_rate),
            'best_avg_reward': float(self.best_avg_reward),

            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'history': self.history
        }, path)
        print(f"Model saved to {path} (Epoch {epoch})")
    
    def load_model(self, path):
        """Load a checkpoint and restore training state."""
        if not os.path.exists(path):
            print(f"[Warning] Model not found: {path}")
            return 0 # Start from beginning if not found

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        try:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        except KeyError:
            print("[Info] Optimizer states not found in checkpoint, skipping.")
        except ValueError:
             print("[Info] Optimizer architecture mismatch (transfer learning?), skipping optimizer load.")

        self.best_success_rate = checkpoint.get('best_success_rate', -1.0)
        self.best_avg_reward = checkpoint.get('best_avg_reward', -float('inf'))
        self.history = checkpoint['history']
        
        start_epoch = checkpoint.get('epoch', 0)
        
        print(f"Resumed from {path} at Epoch {start_epoch}")
        return start_epoch

    @staticmethod
    def apply_transform_static(pos, mode, center_pos):
        """Apply a D4 symmetry transform around the scene center."""
        cx, cy = center_pos[0], center_pos[1]
        rel_x = pos[:, 0] - cx
        rel_y = pos[:, 1] - cy
        
        if mode == 0:   nx, ny = rel_x, rel_y
        elif mode == 1: nx, ny = -rel_y, rel_x
        elif mode == 2: nx, ny = -rel_x, -rel_y
        elif mode == 3: nx, ny = rel_y, -rel_x
        elif mode == 4: nx, ny = -rel_x, rel_y
        elif mode == 5: nx, ny = rel_x, -rel_y
        elif mode == 6: nx, ny = rel_y, rel_x
        elif mode == 7: nx, ny = -rel_y, -rel_x
        else: nx, ny = rel_x, rel_y

        if config_dimension == 3:
            z = pos[:, 2]
            return np.column_stack([nx + cx, ny + cy, z])
            
        return np.column_stack([nx + cx, ny + cy])

    def run_eval_episode(self, idx):
        """Run one validation rollout and return its trajectory metadata."""
        scenario_path = deepcopy(self.expert_data_manager.augmented_trajectories[idx])
        if isinstance(scenario_path, str):
            scenario = np.load(scenario_path, allow_pickle=True).item()
        else:
            scenario = scenario_path

        total_agents = len(scenario['initial_positions'])
        damaged_agents = len(scenario['damaged_indices'])

        env = SwarmEnv(self.expert_data_manager.graph_builder, scenario)
        state = env.reset()
        done = False
        position_history = [env.current_positions.copy()]
        final_info = {} 
        
        while not done:
            if state is None: break
            state_batch = fast_batch_from_dicts([state], self.device)
            self.actor.eval()
            with torch.no_grad():
                actions, _ = self.actor(state_batch, deterministic=True)
            action_np = actions.cpu().numpy()
            state, _, done, info = env.step(action_np)
            position_history.append(env.current_positions.copy())
            final_info = info 

        is_success = final_info.get('connected', False) and (final_info.get('num_subnets', 0) == 1)
        num_subnets = final_info.get('num_subnets', 1)
        
        return {
            'is_success': is_success,
            'steps': env.step_count,
            'positions': position_history,
            'remain': env.remain_indices,
            'damaged': env.damaged_indices,
            'num_subnets': num_subnets,
            'total_agents': total_agents,
            'damaged_agents': damaged_agents,
            'rotation_mode': final_info.get('rotation_mode', 0),
            'original_scenario_idx': final_info.get('original_scenario_idx', -1)
        }

    def validate_visualize(self, epoch):
        """Render selected validation episodes and save redraw metadata."""
        print(f"[Visualization] Validating examples for Epoch {epoch}...")
        
        if run_debug:
            visualize_idx = [0]
        else:
            visualize_idx = [2757, 8207, 15537]

        for idx in visualize_idx:
            result = self.run_eval_episode(idx)
            
            expert_traj_plot = None
            expert_steps = 0
            original_idx = result['original_scenario_idx']
            rotation_mode = result['rotation_mode']
            
            if original_idx in self.expert_data_manager.id_to_traj_map:
                raw_expert_data = self.expert_data_manager.id_to_traj_map[original_idx]

                if 'trajectory' in raw_expert_data:
                    raw_traj = raw_expert_data['trajectory']
                    center_pos = np.array(config_central_point, dtype=np.float32)
                    
                    rotated_traj = []
                    for step_data in raw_traj:
                        pos = step_data['positions']
                        rot_pos = self.apply_transform_static(pos, rotation_mode, center_pos)
                        rotated_traj.append(rot_pos)
                    
                    expert_traj_plot = rotated_traj
                    expert_steps = len(rotated_traj)
            
            fig, axes = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)

            # Left plot: Actor
            status = "Success" if result['is_success'] else "Failed"
            title_actor = f"Validation - Epoch {epoch}\n{status} | Steps: {result['steps']} | Subnets: {result['num_subnets']}"
            if epoch == 0: title_actor = f"Initialization - Epoch {epoch}\nDamaged nodes: {len(result['damaged'])} | Subnets: {result['num_subnets']}"
            self._plot_on_axis(axes, result['positions'], result['remain'], result['damaged'], title_actor)
            
            plt.tight_layout()
            base_name = f"viz_epoch_{epoch}_dmg{len(result['damaged'])}"
            save_path = os.path.join(fig_save_dir, f"{base_name}.png")
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            plt.close(fig)
            self._save_visualization_data(
                save_dir=fig_save_dir,
                base_name=base_name,
                epoch=epoch,
                result=result,
                title=title_actor
            )

    def _save_visualization_data(self, save_dir, base_name, epoch, result, title):
        """Save the raw data needed to redraw a validation visualization."""
        data_save_path = os.path.join(save_dir, f"{base_name}.npz")
        title_lines = title.split('\n')
        title_line_1 = title_lines[0] if len(title_lines) > 0 else ''
        title_line_2 = title_lines[1] if len(title_lines) > 1 else ''

        np.savez_compressed(
            data_save_path,
            epoch=np.array(epoch, dtype=np.int32),
            positions=np.array(result['positions'], dtype=np.float32),
            remain_indices=np.array(result['remain'], dtype=np.int32),
            damaged_indices=np.array(result['damaged'], dtype=np.int32),
            center_point=np.array(config_central_point, dtype=np.float32),
            communication_range=np.array(config_communication_range, dtype=np.float32),
            is_success=np.array(result['is_success'], dtype=bool),
            steps=np.array(result['steps'], dtype=np.int32),
            num_subnets=np.array(result['num_subnets'], dtype=np.int32),
            total_agents=np.array(result['total_agents'], dtype=np.int32),
            damaged_agents=np.array(result['damaged_agents'], dtype=np.int32),
            rotation_mode=np.array(result['rotation_mode'], dtype=np.int32),
            original_scenario_idx=np.array(result['original_scenario_idx'], dtype=np.int32),
            title=np.array(title),
            title_line_1=np.array(title_line_1),
            title_line_2=np.array(title_line_2),
        )
        print(f"[Visualization] Raw redraw data saved to {data_save_path}")
                
    def _plot_on_axis(self, ax, positions_list, remain_indices, damaged_indices, title):
        """Draw one rollout on a Matplotlib axis."""
        pos_array = np.array(positions_list)
        final_pos = pos_array[-1]
        
        for i in remain_indices:
            ax.plot(pos_array[:, i, 0], pos_array[:, i, 1], color='royalblue', alpha=0.3, linewidth=0.8)
            ax.scatter(pos_array[-1, i, 0], pos_array[-1, i, 1], color='navy', s=20)
            
        if len(damaged_indices) > 0:
            ax.scatter(pos_array[0, damaged_indices, 0], pos_array[0, damaged_indices, 1], 
                      color='red', marker='x', s=30, label='Damaged')
            
        # Plot communication connections (only last frame)
        dists = cdist(final_pos[remain_indices], final_pos[remain_indices])
        adj = (dists < config_communication_range) & (dists > 1e-6)
        rows, cols = np.where(np.triu(adj))
        for r, c in zip(rows, cols):
            p1 = final_pos[remain_indices[r]]
            p2 = final_pos[remain_indices[c]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='limegreen', alpha=0.4, linewidth=0.5)
            
        # Center point
        center = config_central_point
        ax.scatter(center[0], center[1], color='green', marker='*', s=150)
        
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
