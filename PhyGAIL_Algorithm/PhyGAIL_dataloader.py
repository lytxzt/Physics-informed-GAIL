import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

from Configurations import *
from PhyGAIL_Algorithm.PhyGAIL_config import *
from PhyGAIL_Algorithm.PhyGAIL_utils import apply_random_augmentation, augment_scenario_entry

# ==============================
# Data Preprocessing and Graph Construction
# ==============================
class SubnetGraphBuilder:
    """Build PhyGAIL graph inputs from global swarm state."""
    
    def __init__(self):
        self.comm_range = config_communication_range
        
    def extract_subnets(self, positions, remain_indices):
        """Extract connected components from the surviving swarm."""
        remain_indices = np.array(remain_indices, dtype=np.int64)
        
        n_remain = len(remain_indices)
        if n_remain == 0: return []
        if n_remain == 1: return [remain_indices]
        
        remain_positions = positions[remain_indices]
        diff = remain_positions[:, np.newaxis, :] - remain_positions[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        comm_range_sq = self.comm_range ** 2
        adj_matrix = dist_sq < comm_range_sq
        graph = csr_matrix(adj_matrix)
        n_components, labels = connected_components(graph, directed=False, return_labels=True)
        subnets = []
        if n_components == 1:
            subnets.append(remain_indices)
        else:
            for i in range(n_components):
                comp_indices = np.where(labels == i)[0]
                subnets.append(remain_indices[comp_indices])
        
        return subnets
    
    def find_damaged_neighbors(self, subnet_indices, positions, damaged_indices):
        """Return damaged nodes visible to the given subnet."""
        if len(damaged_indices) == 0: return []
        
        pos_subnet = positions[subnet_indices]
        pos_damaged = positions[damaged_indices]
        
        dists = cdist(pos_subnet, pos_damaged)
        mask = np.any(dists < self.comm_range, axis=0)
        
        return list(np.array(damaged_indices)[mask])

    def build_graph_dicts(self, positions, velocities, remain_indices, damaged_indices=None,
                          center_pos=None, is_training=False):
        """Build all subnet graph dictionaries for the current swarm state."""
        if center_pos is None:
            center_pos = np.array(config_central_point, dtype=np.float32)
        else:
            center_pos = np.asarray(center_pos, dtype=np.float32)

        remain_indices = np.array(remain_indices, dtype=np.int64)
        if remain_indices.size == 0:
            return []

        if damaged_indices is None:
            total_agents = positions.shape[0]
            remain_mask = np.zeros(total_agents, dtype=bool)
            remain_mask[remain_indices] = True
            damaged_indices = np.where(~remain_mask)[0]
        else:
            damaged_indices = np.array(damaged_indices, dtype=np.int64)

        graphs = []
        for subnet_indices in self.extract_subnets(positions, remain_indices):
            damaged_neighbors = self.find_damaged_neighbors(subnet_indices, positions, damaged_indices)
            graph = self._build_subnet_graph(
                positions,
                velocities,
                subnet_indices,
                damaged_neighbors,
                center_pos,
                is_training=is_training,
            )
            if graph is not None:
                graphs.append(graph)
        return graphs
    
    def _build_subnet_graph(self, positions, velocities, subnet_indices, damaged_neighbors, 
                           center_pos, expert_direction=None, expert_distance=None, 
                           all_remain_indices=None, is_training=True):
        """Build a single subnet graph as a NumPy dictionary."""
        del expert_direction, expert_distance, all_remain_indices, is_training
        
        num_subnet = len(subnet_indices)
        num_damaged = len(damaged_neighbors)
        pos_subnet = positions[subnet_indices]
        pos_damaged = positions[damaged_neighbors] if len(damaged_neighbors) > 0 else np.empty((0, config_dimension))
        node_positions = np.vstack([pos_subnet, pos_damaged, center_pos.reshape(1, config_dimension)])

        vel_subnet = velocities[subnet_indices]
        vel_damaged = np.zeros((num_damaged, config_dimension))
        vel_center = np.zeros((1, config_dimension))
        node_velocities = np.vstack([vel_subnet, vel_damaged, vel_center])
        node_velocities = node_velocities / config_max_speed
        
        num_nodes = num_subnet + num_damaged + 1
        center_idx = num_nodes - 1
        
        node_type = np.zeros(num_nodes, dtype=np.int64)
        if num_damaged > 0:
            node_type[num_subnet : num_subnet + num_damaged] = 1
        node_type[center_idx] = 2
        
        rel_to_center = node_positions - center_pos
        scale_factor = 0.5 * config_width
        normalized_pos = rel_to_center / scale_factor
        
        sources, targets, types = [], [], []
        
        # Communication edges between alive nodes.
        if num_subnet > 1:
            dists_sub = cdist(pos_subnet, pos_subnet)
            
            np.fill_diagonal(dists_sub, np.inf)
            
            K_neighbors = min(8, num_subnet - 1)
            knn_mask = np.zeros_like(dists_sub, dtype=bool)
            
            if K_neighbors > 0:
                knn_indices = np.argsort(dists_sub, axis=1)[:, :K_neighbors]
                np.put_along_axis(knn_mask, knn_indices, True, axis=1)
                
            adj_sub = (dists_sub < self.comm_range) & knn_mask
            
            adj_sub = adj_sub | adj_sub.T
            
            s, t = np.where(adj_sub)
            sources.append(s); targets.append(t); types.append(np.zeros_like(s))
            
            actual_k_act_per_node = adj_sub.sum(axis=1) 
        else:
            actual_k_act_per_node = np.zeros(num_subnet, dtype=int)

        # Observation edges from damaged nodes to alive nodes.
        if num_subnet > 0 and num_damaged > 0:
            dists_vis = cdist(pos_subnet, pos_damaged)
            
            k_dmg_per_node = np.minimum(3, actual_k_act_per_node)
            k_dmg_per_node = np.minimum(k_dmg_per_node, num_damaged)
            
            ranks_vis = np.argsort(np.argsort(dists_vis, axis=1), axis=1)
            
            knn_mask_vis = ranks_vis < k_dmg_per_node[:, np.newaxis]
            
            adj_vis = (dists_vis < self.comm_range) & knn_mask_vis
            
            s, t = np.where(adj_vis)
            sources.append(t + num_subnet); targets.append(s); types.append(np.ones_like(s))

        if num_subnet > 0:
            s = np.full(num_subnet, center_idx, dtype=np.int64)
            t = np.arange(num_subnet, dtype=np.int64)
            sources.append(s); targets.append(t); types.append(np.full(num_subnet, 2, dtype=np.int64))
            
        if not sources:
            all_src = np.array([], dtype=np.int64)
            all_dst = np.array([], dtype=np.int64)
            all_types = np.array([], dtype=np.int64)
        else:
            all_src = np.concatenate(sources)
            all_dst = np.concatenate(targets)
            all_types = np.concatenate(types)
        
        valid_mask = (all_types == 0) | (all_types == 2)
        degrees = np.bincount(all_src[valid_mask], minlength=num_nodes)
        FIXED_MAX_DEGREE = 20.0
        norm_degrees = np.log1p(degrees) / np.log1p(FIXED_MAX_DEGREE)
        norm_degrees[-1] = 0.99
        
        node_feats = np.column_stack([
            normalized_pos,
            node_velocities,
            norm_degrees
        ]).astype(np.float32)

        remain_mask = np.zeros(num_nodes, dtype=bool); remain_mask[:num_subnet] = True
        
        return {
            'x': node_feats,
            'edge_index': np.vstack([all_src, all_dst]),
            'edge_type': all_types,
            'node_type': node_type,
            'remain_mask': remain_mask,
            'subnet_indices': np.array(subnet_indices, dtype=np.int64),
            'num_nodes': num_nodes
        }
    
    def build_from_raw_state(self, state_dict, center_pos):
        """Convert raw trajectory state into one expert graph."""
        positions = state_dict['positions']
        velocities = state_dict['velocities']
        remain_mask = state_dict['remain_mask']
        damaged_mask = state_dict['damaged_mask']
        
        remain_indices = np.where(remain_mask)[0] if remain_mask.dtype==bool else remain_mask
        damaged_indices = np.where(damaged_mask)[0] if damaged_mask.dtype==bool else damaged_mask
        
        damaged_neighbors = self.find_damaged_neighbors(remain_indices, positions, damaged_indices)
        return self._build_subnet_graph(positions, velocities, remain_indices, damaged_neighbors, center_pos, is_training=False)


# ==========================================
# Core Dataset Class
# ==========================================
class LazyExpertDataset(Dataset):
    """Lazy dataset that converts expert trajectories into PyG graphs on demand."""

    def __init__(self, raw_trajectories, graph_builder, step_interval=2, augment=False):
        self.raw_trajectories = raw_trajectories
        self.graph_builder = graph_builder
        self.step_interval = step_interval
        self.augment = augment
        self.center_pos = np.array(config_central_point, dtype=np.float32)
        
        self.indices = []
        for t_idx, traj in enumerate(raw_trajectories):
            if not isinstance(traj, dict) or 'trajectory' not in traj:
                continue
            traj_len = len(traj['trajectory'])
            for s_idx in range(0, traj_len, step_interval):
                self.indices.append((t_idx, s_idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, step_idx = self.indices[idx]
        step_data = self.raw_trajectories[traj_idx]['trajectory'][step_idx]
        
        positions = step_data['positions']
        velocities = step_data['velocities'] / config_dt
        
        if self.augment:
            positions, velocities = apply_random_augmentation(positions, velocities, self.center_pos)
            
        raw_state = {
            'positions': positions,
            'velocities': velocities,
            'remain_mask': step_data['remain_mask'],
            'damaged_mask': step_data['damaged_mask']
        }
        
        graph = self.graph_builder.build_from_raw_state(raw_state, self.center_pos)
        
        if graph is None:
            return self.__getitem__((idx + 1) % len(self))
        
        if isinstance(graph, dict):
            graph = Data(
                x=torch.from_numpy(graph['x']).float(),
                edge_index=torch.from_numpy(graph['edge_index']).long(),
                edge_type=torch.from_numpy(graph['edge_type']).long(),
                node_type=torch.from_numpy(graph['node_type']).long(),
                remain_mask=torch.from_numpy(graph['remain_mask']).bool(),
                subnet_indices=torch.from_numpy(graph['subnet_indices']).long(),
                num_nodes=graph['num_nodes']
            )

        mask = step_data['remain_mask']
        if mask.dtype == bool or isinstance(mask[0], (bool, np.bool_)):
            alive_action = velocities[mask]
        else:
            alive_action = velocities[mask]
            
        graph.expert_action = torch.from_numpy(alive_action).float()
        
        return graph


# ==============================
# Expert Data Manager (Integrated Data Augmentation)
# ==============================
class ExpertDataManager:
    """Lazy-loading expert data manager for PhyGAIL training."""
    def __init__(self, data_source, graph_builder, augmentation_factor=1):
        self.graph_builder = graph_builder
        self.batch_size = 128
        self.num_workers = min(os.cpu_count(), 8)
        
        if isinstance(data_source, str):
            print(f"Loading raw data from {data_source}...")
            raw_data = np.load(data_source, allow_pickle=True)
            if 'trajectories' in raw_data: self.raw_trajs = raw_data['trajectories'].tolist()
            elif 'data' in raw_data: self.raw_trajs = raw_data['data'].tolist()
            else: self.raw_trajs = raw_data['arr_0'].tolist()
        else:
            self.raw_trajs = data_source

        self.id_to_traj_map = {}
        for idx, traj in enumerate(self.raw_trajs):
            self.id_to_traj_map[idx] = traj

        self.augmented_trajectories = []
        center_pos = np.array(config_central_point, dtype=np.float32)
        
        print("Preparing scenario metadata...")
        for traj_idx, traj in enumerate(self.raw_trajs):
            if not isinstance(traj, dict): continue
            
            if augmentation_factor > 1:
                aug_scenarios = augment_scenario_entry(traj, center_pos, global_idx=traj_idx)
                self.augmented_trajectories.extend(aug_scenarios[:augmentation_factor])
            else:
                traj_copy = traj.copy()
                traj_copy['original_scenario_idx'] = traj_idx
                traj_copy['rotation_mode'] = 0
                self.augmented_trajectories.append(traj_copy)

        do_augment = (augmentation_factor > 1)
        self.dataset = LazyExpertDataset(
            self.raw_trajs, 
            graph_builder, 
            step_interval=2, 
            augment=do_augment
        )
        
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=(self.num_workers > 0)
        )
        
        self.iterator = iter(self.loader)
        print(f"Lazy Loader Ready: {len(self.dataset)} frames, using {self.num_workers} workers.")

    def sample_batch(self, batch_size):
        """Return the next expert batch, restarting the loader if needed."""
        if batch_size != self.loader.batch_size:
            pass

        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
            
        return batch
