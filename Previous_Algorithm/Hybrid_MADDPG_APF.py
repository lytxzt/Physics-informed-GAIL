import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import Utils
from Configurations import *

class SimpleMADDPGActor(nn.Module):
    def __init__(self, obs_dim, action_dim=3, hidden_dim=128):
        super(SimpleMADDPGActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, obs):
        return self.net(obs)

class Hybrid_MADDPG_APF:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        obs_dim = config_dimension * 3
        self.actor = SimpleMADDPGActor(obs_dim=obs_dim, action_dim=config_dimension).to(self.device)
        
        self.actor.eval()
        
        self.k_att = 0.8
        self.k_rep = 15.0
        self.safe_distance = 15.0
        self.max_speed = config_constant_speed

    def get_local_observation(self, true_positions, remain_list, i_idx):
        """Build a compact local observation for the baseline actor."""
        pos = true_positions[i_idx]
        vel = np.zeros(config_dimension)
        
        neighbor_diffs = []
        for j_idx in remain_list:
            if i_idx != j_idx:
                dist = np.linalg.norm(true_positions[j_idx] - pos)
                if dist <= config_communication_range:
                    neighbor_diffs.append(true_positions[j_idx] - pos)
                    
        if len(neighbor_diffs) > 0:
            avg_neighbor_diff = np.mean(neighbor_diffs, axis=0)
        else:
            avg_neighbor_diff = np.zeros(config_dimension)
            
        obs = np.concatenate([pos, vel, avg_neighbor_diff])
        return torch.FloatTensor(obs).to(self.device)

    def compute_apf_velocity(self, true_positions, remain_list, subnets, i_idx):
        """Compute the APF-based guidance velocity."""
        pos_i = true_positions[i_idx]
        f_att = np.zeros(config_dimension)
        f_rep = np.zeros(config_dimension)

        my_subnet = None
        for subnet in subnets:
            if i_idx in subnet:
                my_subnet = subnet
                break
                
        if my_subnet is not None and len(subnets) > 1:
            other_nodes = [n for n in remain_list if n not in my_subnet]
            if len(other_nodes) > 0:
                center_of_others = np.mean(true_positions[other_nodes], axis=0)
                diff = center_of_others - pos_i
                dist = np.linalg.norm(diff)
                if dist > 0:
                    f_att = self.k_att * (diff / dist)

        for j_idx in remain_list:
            if i_idx != j_idx:
                diff = pos_i - true_positions[j_idx]
                dist = np.linalg.norm(diff)
                if 0 < dist < self.safe_distance:
                    force_mag = self.k_rep * (1.0/dist - 1.0/self.safe_distance) * (1.0 / (dist**2))
                    f_rep += force_mag * (diff / dist)

        v_apf = f_att + f_rep
        v_mag = np.linalg.norm(v_apf)
        if v_mag > self.max_speed:
            v_apf = (v_apf / v_mag) * self.max_speed
            
        return v_apf

    def get_actions(self, true_positions, remain_list, subnets):
        actions = np.zeros((config_num_of_agents, config_dimension))
        temp_max_distance = 0.0

        for i_idx in remain_list:
            v_apf = self.compute_apf_velocity(true_positions, remain_list, subnets, i_idx)
            
            with torch.no_grad():
                obs = self.get_local_observation(true_positions, remain_list, i_idx)
                v_rl_tensor = self.actor(obs)
                v_rl = v_rl_tensor.cpu().numpy() * self.max_speed
                
            alpha = 0.5
            v_final = alpha * v_rl + (1 - alpha) * v_apf
            
            v_mag = np.linalg.norm(v_final)
            if v_mag > 0:
                actions[i_idx] = v_final / v_mag
                
            temp_max_distance = max(temp_max_distance, v_mag * config_dt)

        max_time = temp_max_distance / config_constant_speed if config_constant_speed > 0 else 0
        return deepcopy(actions), deepcopy(max_time)
