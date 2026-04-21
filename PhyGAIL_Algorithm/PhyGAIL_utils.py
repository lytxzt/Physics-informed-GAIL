import os
import numpy as np
import torch
from torch_geometric.data import Data, Batch
import random
from tqdm import tqdm
import concurrent.futures
import matplotlib.pyplot as plt

from Configurations import *
from PhyGAIL_Algorithm.PhyGAIL_config import *

# ==============================
# Expert Trajectory Generation Functions
# ==============================
def _process_single_scenario(args):
    """
    Process single scenario (for parallelization)
    Uses vectorized computation instead of for-loop simulation
    """
    scenario, idx, max_speed, dt, max_steps = args
    
    initial_positions = scenario['initial_positions'].astype(np.float32)
    remain_indices = scenario['remain_indices']
    damaged_indices = scenario['damaged_indices']
    expert_solution = scenario['expert_solution'].astype(np.float32)
    
    n_agents = len(initial_positions)
    
    # 1. Prepare data
    # Target positions: damaged nodes stay in place, remaining nodes move to expert_solution
    target_positions = initial_positions.copy()
    target_positions[remain_indices] = expert_solution
    
    # Calculate displacement vectors and distances
    total_displacement = target_positions - initial_positions
    total_distances = np.linalg.norm(total_displacement, axis=1)
    
    # Calculate unit direction vectors (avoid division by zero)
    # For nodes with zero distance (damaged nodes), set direction to 0
    with np.errstate(divide='ignore', invalid='ignore'):
        directions = total_displacement / total_distances[:, np.newaxis]
    directions[total_distances == 0] = 0
    
    # 2. Vectorized time step calculation
    # Only remaining nodes need to move
    remain_dists = total_distances[remain_indices]
    if len(remain_dists) > 0:
        steps_needed = np.ceil(remain_dists / (max_speed * dt)).astype(int)
        actual_steps = min(np.max(steps_needed) + 1, max_steps)
    else:
        actual_steps = 2  # Only damaged nodes, generate a few frames
        
    # Generate time array
    time_steps = np.arange(actual_steps) * dt
    time_steps = time_steps.reshape(-1, 1)
    
    # 3. Broadcast calculation of positions at all time steps
    dist_traveled = time_steps * max_speed
    dist_traveled = np.repeat(dist_traveled, n_agents, axis=1)
    dist_traveled = np.minimum(dist_traveled, total_distances)
    
    # Position calculation: P = P0 + dir * dist
    current_positions_over_time = initial_positions[np.newaxis, :, :] + \
                                  dist_traveled[:, :, np.newaxis] * directions[np.newaxis, :, :]
                                  
    # 4. Broadcast calculation of velocities
    is_moving = dist_traveled < (total_distances - 1e-4)
    step_displacement_magnitude = max_speed * dt
    velocities_over_time = is_moving[:, :, np.newaxis] * (directions[np.newaxis, :, :] * step_displacement_magnitude)
    
    # 5. Assemble result list (only loop needed for assembly)
    scenario_trajectory = []
    
    # Precompute masks
    remain_mask = np.isin(np.arange(n_agents), remain_indices)
    damaged_mask = np.isin(np.arange(n_agents), damaged_indices)
    
    for t in range(actual_steps):
        state_data = {
            'positions': current_positions_over_time[t].astype(np.float32),
            'velocities': velocities_over_time[t].astype(np.float32),
            'damaged_mask': damaged_mask,
            'remain_mask': remain_mask
        }
        scenario_trajectory.append(state_data)
        
    # Add final state (ensure perfect alignment)
    final_state = {
        'positions': target_positions.astype(np.float32),
        'velocities': np.zeros((n_agents, config_dimension), dtype=np.float32),
        'damaged_mask': damaged_mask,
        'remain_mask': remain_mask
    }
    scenario_trajectory.append(final_state)
    
    return {
        'initial_positions': initial_positions,
        'damaged_indices': damaged_indices,
        'remain_indices': remain_indices,
        'target_positions': target_positions,
        'trajectory': scenario_trajectory,
        'num_steps': len(scenario_trajectory),
        'scenario_idx': idx
    }


def generate_expert_trajectories(max_speed=10.0, dt=0.1):
    """Parallel generation of expert trajectories (optimized version)"""
    if not os.path.exists(data_path):
        print(f"Error: data file not found at {data_path}")
        return []
    
    expert_data = np.load(data_path, allow_pickle=True)['data'].tolist()
    print(f"Loading {len(expert_data)} expert scenarios")
    
    # Prepare tasks for parallel processing
    tasks = []
    for idx, scenario in enumerate(expert_data):
        tasks.append((scenario, idx, max_speed, dt, config_maximum_step))
    
    # Use process pool for parallel processing
    num_workers = min(os.cpu_count(), 16) 
    print(f"Generating trajectories using {num_workers} CPU cores...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(_process_single_scenario, tasks), 
                           total=len(tasks), desc="Parallel Generation"))
        
    # Filter out any failed results
    trajectories = [r for r in results if r is not None]
    
    print(f"Generated {len(trajectories)} expert trajectories")
    print(f"Average trajectory length: {np.mean([t['num_steps'] for t in trajectories]):.2f}")
    
    return trajectories

# ==========================================
# Helper Functions: On-the-fly Augmentation
# ==========================================
def augment_scenario_entry(traj_data, center_pos, global_idx):
    """
    Apply geometric augmentations to a scenario
    Adds rotation_mode and original_idx recording
    """
    augmented_scenarios = []
    
    init_pos = traj_data['initial_positions']
    has_target = 'expert_solution' in traj_data
    target_pos = traj_data.get('expert_solution', init_pos)
    
    original_idx = global_idx
    cx, cy = center_pos[0], center_pos[1]
    
    def apply_transform(pos, mode):
        """Apply transformation based on symmetry mode"""
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

    # Generate 8 variants (D4 symmetry group)
    for mode in range(8):
        new_scenario = {
            'initial_positions': apply_transform(init_pos, mode).astype(np.float32),
            'damaged_indices': traj_data['damaged_indices'],
            'remain_indices': traj_data['remain_indices'],
            'num_steps': traj_data['num_steps'],
            'rotation_mode': mode,
            'original_scenario_idx': original_idx
        }
        if has_target:
            new_scenario['expert_solution'] = apply_transform(target_pos, mode).astype(np.float32)
            
        augmented_scenarios.append(new_scenario)
        
    return augmented_scenarios

def apply_random_augmentation(pos, vel, center_pos):
    """
    Apply random D4 symmetry transformation to current frame
    """
    mode = random.randint(0, 7)
    if mode == 0: return pos, vel  # Identity transformation
    
    cx, cy = center_pos[0], center_pos[1]
    rel_x = pos[:, 0] - cx
    rel_y = pos[:, 1] - cy
    vx, vy = vel[:, 0], vel[:, 1]
    
    # Transformation logic for D4 symmetry group
    if mode == 1:   # Rotate 90 degrees
        nx, ny, nvx, nvy = -rel_y, rel_x, -vy, vx
    elif mode == 2: # Rotate 180 degrees
        nx, ny, nvx, nvy = -rel_x, -rel_y, -vx, -vy
    elif mode == 3: # Rotate 270 degrees
        nx, ny, nvx, nvy = rel_y, -rel_x, vy, -vx
    elif mode == 4: # Flip X axis
        nx, ny, nvx, nvy = -rel_x, rel_y, -vx, vy
    elif mode == 5: # Flip Y axis
        nx, ny, nvx, nvy = rel_x, -rel_y, vx, -vy
    elif mode == 6: # Flip along diagonal
        nx, ny, nvx, nvy = rel_y, rel_x, vy, vx
    elif mode == 7: # Flip along anti-diagonal
        nx, ny, nvx, nvy = -rel_y, -rel_x, -vy, -vx
    
    if config_dimension == 3:
        z = pos[:, 2]
        vz = vel[:, 2]
        new_pos = np.column_stack([nx + cx, ny + cy, z])
        new_vel = np.column_stack([nvx, nvy, vz])
    else:
        new_pos = np.column_stack([nx + cx, ny + cy])
        new_vel = np.column_stack([nvx, nvy])
    
    return new_pos.astype(np.float32), new_vel.astype(np.float32)


def fast_to_data_list(batch):
    """
    Optimized version: manually split Batch back into Data list
    Avoids RuntimeError from batch.to_data_list()
    """
    data_list = []
    num_graphs = batch.num_graphs
    
    for i in range(num_graphs):
        data = Data()
        
        # 1. Split node attributes
        n_start, n_end = batch.node_ptr[i], batch.node_ptr[i+1]
        data.x = batch.x[n_start:n_end]
        data.node_type = batch.node_type[n_start:n_end]
        data.remain_mask = batch.remain_mask[n_start:n_end]
        data.num_nodes = (n_end - n_start).item()
        
        # 2. Split edge attributes
        e_start, e_end = batch.edge_ptr[i], batch.edge_ptr[i+1]
        if e_start < e_end:
            # Key: subtract node offset to convert to local indices
            data.edge_index = batch.edge_index[:, e_start:e_end] - n_start
            data.edge_type = batch.edge_type[e_start:e_end]
        else:
            # Handle graphs with no edges
            data.edge_index = torch.empty((2, 0), dtype=torch.long, device=batch.x.device)
            data.edge_type = torch.empty((0,), dtype=torch.long, device=batch.x.device)

        # 3. Split subnet indices
        s_start, s_end = batch.subnet_ptr[i], batch.subnet_ptr[i+1]
        data.subnet_indices = batch.subnet_indices[s_start:s_end]
        
        data_list.append(data)
        
    return data_list

def fast_batch_from_dicts(state_dicts, device):
    """
    Optimized Batch construction with debug features and pointer recording
    """
    # 1. Filter invalid data
    valid_dicts = [s for s in state_dicts if s is not None]
    if not valid_dicts: return None

    # Preallocate lists
    xs = []
    edge_indices = []
    edge_types = []
    node_types = []
    remain_masks = []
    subnet_indices_list = []
    batch_vec_list = []
    
    # Record counts for pointer calculation
    node_counts = []
    edge_counts = []
    subnet_counts = []
    
    node_offset = 0
    
    for i, s in enumerate(valid_dicts):
        # Debug protection
        if isinstance(s, str):
            print(f"\n[CRITICAL ERROR] Worker returned an error string: {s}")
            raise RuntimeError("Worker process failed.")

        n = s['num_nodes']
        e = s['edge_index'].shape[1] if (s['edge_index'].ndim > 1 and s['edge_index'].shape[0] > 0) else 0
        sub_n = len(s['subnet_indices'])
        
        # Record counts
        node_counts.append(n)
        edge_counts.append(e)
        subnet_counts.append(sub_n)
        
        # Collect data
        xs.append(s['x'])
        if e > 0:
            edge_indices.append(s['edge_index'] + node_offset)
            edge_types.append(s['edge_type'])
            
        node_types.append(s['node_type'])
        remain_masks.append(s['remain_mask'])
        subnet_indices_list.append(s['subnet_indices'])
        batch_vec_list.append(np.full(n, i, dtype=np.int64))
        
        node_offset += n

    # 2. Fast numpy concatenation
    x_np = np.concatenate(xs, axis=0)
    batch_np = np.concatenate(batch_vec_list, axis=0)
    node_type_np = np.concatenate(node_types, axis=0)
    remain_mask_np = np.concatenate(remain_masks, axis=0)
    subnet_indices_np = np.concatenate(subnet_indices_list, axis=0)
    
    # 3. Convert to Tensor and move to device
    batch = Batch()
    batch.x = torch.from_numpy(x_np).float().to(device)
    batch.batch = torch.from_numpy(batch_np).long().to(device)
    batch.node_type = torch.from_numpy(node_type_np).long().to(device)
    batch.remain_mask = torch.from_numpy(remain_mask_np).bool().to(device)
    batch.subnet_indices = torch.from_numpy(subnet_indices_np).long().to(device)
    batch.num_graphs = len(valid_dicts)
    
    if len(edge_indices) > 0:
        batch.edge_index = torch.from_numpy(np.concatenate(edge_indices, axis=1)).long().to(device)
        batch.edge_type = torch.from_numpy(np.concatenate(edge_types, axis=0)).long().to(device)
    else:
        batch.edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        batch.edge_type = torch.empty((0,), dtype=torch.long, device=device)

    # 4. Calculate and attach split pointers for fast unpacking
    nc = torch.tensor(node_counts, device=device, dtype=torch.long)
    ec = torch.tensor(edge_counts, device=device, dtype=torch.long)
    sc = torch.tensor(subnet_counts, device=device, dtype=torch.long)
    
    batch.node_ptr = torch.cat([torch.zeros(1, device=device, dtype=torch.long), torch.cumsum(nc, 0)])
    batch.edge_ptr = torch.cat([torch.zeros(1, device=device, dtype=torch.long), torch.cumsum(ec, 0)])
    batch.subnet_ptr = torch.cat([torch.zeros(1, device=device, dtype=torch.long), torch.cumsum(sc, 0)])
    
    return batch


def moving_average(data, window_size):
    """Compute the moving average of a 1D array."""
    if len(data) < window_size:
        return data
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

def plot_training_curves(npz_file_path, save_path, window_size=50):
    """Plot training curves from a saved ``training_history.npz`` file."""
    data = np.load(npz_file_path)
    
    epochs = data['epoch']
    actor_loss = data['actor_loss']
    critic_loss = data['critic_loss']
    disc_loss = data['discriminator_loss']
    
    reward = data['reward']
    success_rate = data['success_rate']
    
    val_epoch = data['val_epoch']
    val_reward = data['val_reward']
    val_success_rate = data['val_success_rate']
    
    ma_epochs = epochs[window_size - 1:]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('PPO+PhyGAIL Training & Validation Curves', fontsize=18, fontweight='bold', y=0.98)
    
    ax = axes[0, 0]
    ax.plot(epochs, reward, color='skyblue', alpha=0.3, label='Train (Raw)')
    ax.plot(ma_epochs, moving_average(reward, window_size), color='blue', linewidth=2, label=f'Train MA ({window_size})')
    ax.plot(val_epoch, val_reward, color='darkorange', marker='o', markersize=3, linewidth=1.5, label='Validation')
    
    ax.set_title('Episode Reward')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, success_rate, color='lightgreen', alpha=0.3, label='Train (Raw)')
    ax.plot(ma_epochs, moving_average(success_rate, window_size), color='green', linewidth=2, label=f'Train MA ({window_size})')
    ax.plot(val_epoch, val_success_rate, color='red', marker='o', markersize=3, linewidth=1.5, label='Validation')
    
    ax.set_title('Success Rate')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(epochs, disc_loss, color='purple', linewidth=2)
    ax.set_title('Discriminator Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, actor_loss, color='red', linewidth=2)
    ax.set_title('Actor Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(epochs, critic_loss, color='olive', linewidth=2)
    ax.set_title('Critic Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)

    axes[0, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(save_path, dpi=300)
    print(f"Plot successfully saved to {save_path}")
