import numpy as np
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from tqdm import tqdm
from scipy.spatial.distance import cdist

from Configurations import *
from PhyGAIL_Algorithm.PhyGAIL_config import *

# ==============================
# Environment Wrapper Class (Optimized Version)
# ==============================
class SwarmEnv:
    """Pure NumPy swarm environment."""
    def __init__(self, graph_builder, data_source):
        self.graph_builder = graph_builder
        if isinstance(data_source, dict):
            self.mode = 'fixed'
            self.fixed_scenario = data_source
            self.expert_data_manager = None
        else:
            self.mode = 'random'
            self.expert_data_manager = data_source
            self.fixed_scenario = None
            
        self.max_steps = config_maximum_step if not run_debug else 100
        self.dt = config_dt
        self.max_speed = config_max_speed
        self.reset()

    def reset(self):
        if self.mode == 'fixed':
            scenario = self.fixed_scenario
        else:
            num = len(self.expert_data_manager.augmented_trajectories)
            item = self.expert_data_manager.augmented_trajectories[np.random.randint(num)]
            scenario = np.load(item, allow_pickle=True).item() if isinstance(item, str) else item
            
        self.initial_positions = scenario['initial_positions'].astype(np.float32)
        num_agents = len(self.initial_positions)
        self.current_velocities = np.zeros((num_agents, config_dimension), dtype=np.float32)
        self.damaged_indices = np.array(scenario['damaged_indices'], dtype=np.int64)
        self.remain_indices = np.array(scenario['remain_indices'], dtype=np.int64)
        self.expert_steps = scenario['num_steps'] * 1.2
        
        self.rotation_mode = scenario.get('rotation_mode', 0)
        self.original_scenario_idx = scenario.get('original_scenario_idx', scenario.get('scenario_idx', -1))
        
        self.current_positions = self.initial_positions.copy()
        self.step_count = 0
        self.done = False
        
        self.subnets = self.graph_builder.extract_subnets(self.current_positions, self.remain_indices)
        self.prev_num_subnets = len(self.subnets)
        
        return self._build_env_state_dict()

    def step(self, actions):
        """Advance the environment by one control step."""
        alive_idx = self.remain_indices
        
        # 1. Physics update
        speeds = np.linalg.norm(actions, axis=1, keepdims=True)
        scale = np.where(speeds > self.max_speed, self.max_speed / (speeds + 1e-8), 1.0)
        actions *= scale
        
        # Actions follow the previous state's subnet node ordering.
        if self.current_indices is not None and len(self.current_indices) > 0:
            self.current_positions[self.current_indices] += actions * self.dt
            self.current_velocities.fill(0)
            self.current_velocities[self.current_indices] = actions
        
        self.step_count += 1
        if self.step_count >= self.max_steps: self.done = True
        
        # 2. Connectivity analysis (Scipy)
        self.subnets = self.graph_builder.extract_subnets(self.current_positions, self.remain_indices)
        num_subnets = len(self.subnets)
        connected = (num_subnets == 1)
        subnet_factor = np.sqrt(20/config_num_of_agents)
        
        num_nodes = len(self.current_positions)
        node_rewards = np.zeros(num_nodes, dtype=np.float32)

        comp_step = np.zeros(num_nodes, dtype=np.float32)
        comp_collision = np.zeros(num_nodes, dtype=np.float32)
        comp_terminal = np.zeros(num_nodes, dtype=np.float32)
        
        step_penalty = -5 / config_maximum_step
        node_rewards[alive_idx] += step_penalty
        comp_step[alive_idx] += step_penalty

        curr_pos = self.current_positions[alive_idx]
        r_repel = self.compute_repulsion_reward(curr_pos, safe_dist=15.0, weight=0.01)
        node_rewards[alive_idx] += r_repel
        comp_collision[alive_idx] += r_repel

        self.prev_num_subnets = num_subnets
        
        is_success = connected and (num_subnets == 1)
        
        if is_success:
            self.done = True
            
            base_reward = base_success_reward
            speed_bonus_pool = base_time_reward
            decay_sensitivity = 1.0
            
            safe_expert_steps = max(self.expert_steps, 1e-6)
            
            time_factor = np.exp(decay_sensitivity * (1.0 - self.step_count / safe_expert_steps))
            time_factor = min(time_factor, 3.0)
            
            final_reward = base_reward + speed_bonus_pool * time_factor
            
            node_rewards[alive_idx] += final_reward
            comp_terminal[alive_idx] += final_reward
            
        elif self.done:
            # Apply the failure penalty to surviving nodes only.
            node_rewards[alive_idx] -= base_subnet_penalty * num_subnets
            comp_terminal[alive_idx] -= base_subnet_penalty * num_subnets
        
        # 4. Build next frame state
        next_state_dict = self._build_env_state_dict()
        
        valid_len = len(alive_idx) if len(alive_idx) > 0 else 1
        reward_breakdown = {
            'step': np.sum(comp_step[alive_idx]) / valid_len,
            'collision': np.sum(comp_collision[alive_idx]) / valid_len,
            'terminal': np.sum(comp_terminal[alive_idx]) / valid_len
        }
        
        info = {
            'connected': connected, 
            'num_subnets': num_subnets,
            'rotation_mode': getattr(self, 'rotation_mode', 0),
            'original_scenario_idx': getattr(self, 'original_scenario_idx', -1),
            'reward_breakdown': reward_breakdown,
        }
        return next_state_dict, node_rewards, self.done, info

    def _build_env_state_dict(self):
        """Build the batched graph state for the next policy step."""
        if len(self.subnets) == 0:
            self.current_indices = np.array([], dtype=np.int64)
            return None

        graphs = self.graph_builder.build_graph_dicts(
            self.current_positions,
            self.current_velocities,
            self.remain_indices,
            damaged_indices=self.damaged_indices,
            center_pos=np.array(config_central_point, dtype=np.float32),
            is_training=True,
        )
            
        if not graphs: 
            self.current_indices = np.array([], dtype=np.int64)
            return None
            
        total_nodes = sum(g['num_nodes'] for g in graphs)
        x_list, edge_idx_list, edge_type_list = [], [], []
        node_type_list, remain_mask_list, subnet_idx_list = [], [], []
        
        node_offset = 0
        for g in graphs:
            x_list.append(g['x'])
            if g['edge_index'].shape[1] > 0:
                edge_idx_list.append(g['edge_index'] + node_offset)
                edge_type_list.append(g['edge_type'])
                
            node_type_list.append(g['node_type'])
            remain_mask_list.append(g['remain_mask'])
            subnet_idx_list.append(g['subnet_indices'])
            node_offset += g['num_nodes']
            
        state_dict = {
            'x': np.concatenate(x_list, axis=0),
            'node_type': np.concatenate(node_type_list, axis=0),
            'remain_mask': np.concatenate(remain_mask_list, axis=0),
            'subnet_indices': np.concatenate(subnet_idx_list, axis=0),
            'num_nodes': total_nodes
        }
        
        if edge_idx_list:
            state_dict['edge_index'] = np.concatenate(edge_idx_list, axis=1)
            state_dict['edge_type'] = np.concatenate(edge_type_list, axis=0)
        else:
            state_dict['edge_index'] = np.empty((2, 0), dtype=np.int64)
            state_dict['edge_type'] = np.empty((0,), dtype=np.int64)
            
        self.current_indices = state_dict['subnet_indices']
        return state_dict
    
    def compute_repulsion_reward(self, positions, safe_dist=15.0, weight=0.01):
        """Pure physical collision-avoidance penalty based on pairwise distance violation."""
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dist_matrix, np.inf)

        violation = np.maximum(0, safe_dist - dist_matrix)
        node_repulsion = np.sum(violation, axis=1)
        return -weight * node_repulsion

# ==============================================================================
# Parallel Environment Worker Process
# ==============================================================================
class Worker(Process):
    """Parallel environment worker handling communication with the main process"""
    
    def __init__(self, remote, parent_remote, env_fn_wrapper, seed=None):
        super().__init__()
        self.remote = remote
        self.parent_remote = parent_remote
        self.env = env_fn_wrapper()
        self.seed = seed
        self.daemon = True
        
    def run(self):
        self.parent_remote.close()
        
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        
        while True:
            try:
                cmd, data = self.remote.recv()
                
                if cmd == 'step':
                    next_state, reward, done, info = self.env.step(data)
                    
                    if done:
                        info['final_stat'] = {
                            'connected': info.get('connected', False),
                            'num_subnets': info.get('num_subnets', 1)
                        }
                        next_state = self.env.reset()
                    
                    self.remote.send((None, next_state, reward, done, info))
                    
                elif cmd == 'reset':
                    state = self.env.reset()
                    self.remote.send(state)
                    
                elif cmd == 'get_action_dim':
                    state = self.env.reset()
                    action_dim = len(state['subnet_indices']) if state is not None else 0
                    self.remote.send(action_dim)
                    
                elif cmd == 'close':
                    self.remote.close()
                    break
                    
                else:
                    raise NotImplementedError(f"Command {cmd} not implemented")
                    
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                print(f"Worker Error: {e}")
                break

# ==============================================================================
# Parallel Environment Container
# ==============================================================================
class ParallelEnv:
    """Manager for multiple parallel environments"""
    
    def __init__(self, env_fns, device='cpu', base_seed=3407):
        self.device = torch.device(device)
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.ps = []
        
        for i, (work_remote, remote, env_fn) in enumerate(tqdm(zip(self.work_remotes, self.remotes, env_fns), desc="Generate parallel envs")):
            p = Worker(work_remote, remote, env_fn, seed=base_seed + i)
            self.ps.append(p)
            p.start()
            work_remote.close()
            
        self.action_dims = []
        for remote in self.remotes:
            remote.send(('get_action_dim', None))
            self.action_dims.append(remote.recv())
    
    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        
        results = [remote.recv() for remote in self.remotes]
        next_state_dicts, rewards, dones, infos = zip(*[r[1:] for r in results])
        return list(next_state_dicts), list(rewards), list(dones), list(infos)
    
    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        
        state_dicts = [remote.recv() for remote in self.remotes]
        return state_dicts
    
    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
