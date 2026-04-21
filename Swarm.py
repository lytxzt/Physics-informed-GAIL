from copy import deepcopy
import os

import numpy as np
import torch

import Utils
from Configurations import *
from Environment import Environment
from PhyGAIL_Algorithm.PhyGAIL_config import device
from PhyGAIL_Algorithm.PhyGAIL_dataloader import SubnetGraphBuilder
from PhyGAIL_Algorithm.PhyGAIL_network import ActorNetwork
from PhyGAIL_Algorithm.PhyGAIL_utils import fast_batch_from_dicts
from Previous_Algorithm.Centering import centering_fly_v2
from Previous_Algorithm.CR_MGC import CR_MGC
from Previous_Algorithm.DEMD import DEMD
from Previous_Algorithm.GDR_TS import GDR_TS
from Previous_Algorithm.HERO import HERO
from Previous_Algorithm.Hybrid_MADDPG_APF import Hybrid_MADDPG_APF
from Previous_Algorithm.SIDR import SIDR


PREVIOUS_ALGORITHM_ORDER = [
    "centering",
    "HERO",
    "SIDR",
    "CR-MGC",
    "DEMD",
    "GDR-TS",
    "MADDPG-APF",
]

ALGORITHM_NAME_TO_MODE = {
    "centering": 1,
    "HERO": 2,
    "SIDR": 3,
    "CR-MGC": 4,
    "DEMD": 5,
    "GDR-TS": 6,
    "MADDPG-APF": 7,
    "PhyGAIL": 11,
}

ALGORITHM_MODE_TO_NAME = {value: key for key, value in ALGORITHM_NAME_TO_MODE.items()}


class Swarm:
    """Swarm simulator wrapper with PhyGAIL and baseline controllers."""

    def __init__(self, algorithm_mode="PhyGAIL", use_pretrained=False, khop=3, initial_positions=None,
                 model_path="models/model_best.pth"):
        del use_pretrained
        if initial_positions is not None:
            self.initial_positions = deepcopy(initial_positions)
        else:
            self.initial_positions = deepcopy(config_initial_swarm_positions)

        self.num_of_agents = config_num_of_agents
        self.remain_list = [i for i in range(config_num_of_agents)]
        self.remain_num = config_num_of_agents
        self.max_destroy_num = config_maximum_destroy_num

        self.remain_positions = deepcopy(self.initial_positions)
        self.true_positions = deepcopy(self.initial_positions)
        self.current_velocities = np.zeros((self.num_of_agents, config_dimension), dtype=np.float32)

        self.khop = khop
        self.model_path = model_path
        self.device = device
        self.algorithm_mode = self._normalize_algorithm_mode(algorithm_mode)
        self.algorithm_name = ALGORITHM_MODE_TO_NAME[self.algorithm_mode]

        self.if_once_network = False
        self.once_destroy_speed = np.zeros((self.num_of_agents, config_dimension))
        self.best_final_positions = 0
        self.max_time = 0

        self.graph_builder = SubnetGraphBuilder()
        self._init_algorithms()

    def _normalize_algorithm_mode(self, algorithm_mode):
        """Normalize a user-facing algorithm name or numeric mode."""
        if isinstance(algorithm_mode, str):
            if algorithm_mode not in ALGORITHM_NAME_TO_MODE:
                raise ValueError(f"Unknown algorithm name: {algorithm_mode}")
            return ALGORITHM_NAME_TO_MODE[algorithm_mode]
        if algorithm_mode not in ALGORITHM_MODE_TO_NAME:
            raise ValueError(f"Unknown algorithm mode: {algorithm_mode}")
        return algorithm_mode

    def _init_algorithms(self):
        """Instantiate the controller associated with the current mode."""
        if self.algorithm_mode == 2:
            self.hero = HERO(self.initial_positions)
        elif self.algorithm_mode == 4:
            self.cr_mgc = CR_MGC(use_meta=False)
        elif self.algorithm_mode == 5:
            self.demd = DEMD()
        elif self.algorithm_mode == 6:
            self.gdr_ts = GDR_TS()
        elif self.algorithm_mode == 7:
            self.maddpg_apf = Hybrid_MADDPG_APF()
        elif self.algorithm_mode == 11:
            self.actor = ActorNetwork(action_dim=config_dimension, hidden_dim=128).to(self.device)
            self.model_loaded = False
            self._load_phygail_model()

    def _load_phygail_model(self):
        """Load the released PhyGAIL policy checkpoint."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"PhyGAIL model not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if "actor_state_dict" in checkpoint:
            self.actor.load_state_dict(checkpoint["actor_state_dict"])
        else:
            self.actor.load_state_dict(checkpoint)
        self.actor.eval()
        self.model_loaded = True

    def reset(self, change_algorithm_mode=False, algorithm_mode=None):
        """Reset swarm state and optionally switch controller."""
        self.remain_list = [i for i in range(self.num_of_agents)]
        self.remain_num = self.num_of_agents
        self.true_positions = deepcopy(self.initial_positions)
        self.remain_positions = deepcopy(self.initial_positions)
        self.current_velocities = np.zeros((self.num_of_agents, config_dimension), dtype=np.float32)

        self.if_once_network = False
        self.once_destroy_speed = np.zeros((self.num_of_agents, config_dimension))
        self.best_final_positions = 0
        self.max_time = 0

        if change_algorithm_mode:
            self.algorithm_mode = self._normalize_algorithm_mode(algorithm_mode)
            self.algorithm_name = ALGORITHM_MODE_TO_NAME[self.algorithm_mode]
            self._init_algorithms()

    def destroy_happens(self, destroy_list, environment_positions):
        """Apply a damage event and update surviving nodes."""
        for destroy_index in destroy_list:
            self.remain_list.remove(destroy_index)
        self.true_positions = deepcopy(environment_positions)
        self.remain_num = len(self.remain_list)
        self.make_remain_positions()

    def update_true_positions(self, environment_positions):
        """Synchronize internal positions with the external environment."""
        self.true_positions = deepcopy(environment_positions)

    def make_remain_positions(self):
        """Cache the positions of surviving nodes."""
        self.remain_positions = np.array([deepcopy(self.true_positions[i]) for i in self.remain_list])

    def check_number_of_clusters(self):
        """Return the Laplacian spectrum and component count of surviving nodes."""
        m, _ = self.remain_positions.shape
        G = np.matmul(self.remain_positions, self.remain_positions.T)
        H = np.tile(np.diag(G), (m, 1))
        D = np.sqrt(H + H.T - 2 * G)
        A = np.where(D > config_communication_range, 0, 1.0)
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        e_vals, _ = np.linalg.eigh(L)
        num = np.sum(np.where(e_vals.real < 0.000001, 1, 0))
        return e_vals.real, num

    def _take_phygail_actions(self):
        """Run a deterministic PhyGAIL policy step."""
        actions = np.zeros((self.num_of_agents, config_dimension), dtype=np.float32)
        remain_mask = np.zeros(self.num_of_agents, dtype=bool)
        remain_mask[self.remain_list] = True
        damaged_indices = np.where(~remain_mask)[0]
        graphs = self.graph_builder.build_graph_dicts(
            self.true_positions.astype(np.float32),
            self.current_velocities.astype(np.float32),
            self.remain_list,
            damaged_indices=damaged_indices,
            center_pos=np.array(config_central_point, dtype=np.float32),
            is_training=False,
        )

        if not graphs:
            return actions, 0

        batch = fast_batch_from_dicts(graphs, self.device)
        with torch.no_grad():
            network_actions, _ = self.actor(batch, deterministic=True)

        flat_actions = network_actions.cpu().numpy()
        flat_indices = batch.subnet_indices.cpu().numpy()
        if len(flat_actions) != len(flat_indices):
            raise RuntimeError("PhyGAIL output size does not match subnet index size.")

        actions[flat_indices] = flat_actions * config_dt
        return actions, 0

    def take_actions(self):
        """Dispatch one control step to the active controller."""
        actions = np.zeros((self.num_of_agents, config_dimension), dtype=np.float32)
        max_time = 0
        self.make_remain_positions()
        flag, _ = Utils.check_if_a_connected_graph(deepcopy(self.remain_positions), len(self.remain_list))
        if flag:
            return actions, max_time

        if self.algorithm_mode == 1:
            actions = centering_fly_v2(self.true_positions, self.remain_list)
        elif self.algorithm_mode == 2:
            destroy_index = Utils.difference_set([i for i in range(self.num_of_agents)], self.remain_list)
            actions = self.hero.hero(destroy_index, self.true_positions)
        elif self.algorithm_mode == 3:
            actions = SIDR(self.true_positions, self.remain_list)
        elif self.algorithm_mode == 4:
            if self.if_once_network:
                for i in range(len(self.remain_list)):
                    remain_idx = self.remain_list[i]
                    if np.linalg.norm(self.true_positions[remain_idx] - self.best_final_positions[i]) >= 0.55:
                        actions[remain_idx] = deepcopy(self.once_destroy_speed[remain_idx])
                max_time = deepcopy(self.max_time)
            else:
                self.if_once_network = True
                actions, max_time, best_final_positions = self.cr_mgc.cr_gcm(
                    deepcopy(self.true_positions), deepcopy(self.remain_list)
                )
                self.once_destroy_speed = deepcopy(actions)
                self.best_final_positions = deepcopy(best_final_positions)
                self.max_time = deepcopy(max_time)
        elif self.algorithm_mode == 5:
            if self.if_once_network:
                for i in range(len(self.remain_list)):
                    remain_idx = self.remain_list[i]
                    distance = np.linalg.norm(self.true_positions[remain_idx] - self.best_final_positions[i])
                    if distance >= 1:
                        actions[remain_idx] = deepcopy(self.once_destroy_speed[remain_idx])
                    elif distance > 0.0001:
                        actions[remain_idx] = deepcopy(self.best_final_positions[i] - self.true_positions[remain_idx])
                max_time = deepcopy(self.max_time)
            else:
                self.if_once_network = True
                actions, max_time, best_final_positions = self.demd.demd_adaptive(
                    deepcopy(self.true_positions), deepcopy(self.remain_list)
                )
                self.once_destroy_speed = deepcopy(actions)
                self.best_final_positions = deepcopy(best_final_positions)
                self.max_time = deepcopy(max_time)
        elif self.algorithm_mode == 6:
            if self.if_once_network:
                for i in range(len(self.remain_list)):
                    remain_idx = self.remain_list[i]
                    if np.linalg.norm(self.true_positions[remain_idx] - self.best_final_positions[i]) >= 0.55:
                        actions[remain_idx] = deepcopy(self.once_destroy_speed[remain_idx])
                max_time = deepcopy(self.max_time)
            else:
                self.if_once_network = True
                actions, max_time, best_final_positions = self.gdr_ts.gdr_ts(
                    deepcopy(self.true_positions), deepcopy(self.remain_list)
                )
                self.once_destroy_speed = deepcopy(actions)
                self.best_final_positions = deepcopy(best_final_positions)
                self.max_time = deepcopy(max_time)
        elif self.algorithm_mode == 7:
            subnets = self.graph_builder.extract_subnets(self.true_positions, self.remain_list)
            actions, max_time = self.maddpg_apf.get_actions(
                deepcopy(self.true_positions), deepcopy(self.remain_list), subnets
            )
            self.max_time = deepcopy(max_time)
        elif self.algorithm_mode == 11:
            actions, max_time = self._take_phygail_actions()
        else:
            raise ValueError(f"Unsupported algorithm mode: {self.algorithm_mode}")

        self.current_velocities = deepcopy(actions / config_dt)
        return deepcopy(actions), deepcopy(max_time)

    def solve(self):
        """Run a full recovery episode until success or timeout."""
        max_step = config_maximum_step
        connected_flag = False
        trajectory_log = []
        total_collisions = 0
        num = None

        for step in range(max_step):
            actions, _ = self.take_actions()
            real_velocities = actions * config_constant_speed

            remain_mask = np.zeros(self.num_of_agents, dtype=bool)
            remain_mask[self.remain_list] = True
            damaged_mask = ~remain_mask
            trajectory_log.append(
                {
                    "positions": deepcopy(self.true_positions).astype(np.float32),
                    "velocities": deepcopy(real_velocities).astype(np.float32),
                    "remain_mask": remain_mask,
                    "damaged_mask": damaged_mask,
                }
            )

            positions = self.true_positions + real_velocities
            self.update_true_positions(positions)
            self.make_remain_positions()

            if len(self.remain_positions) > 1:
                diff = self.remain_positions[:, np.newaxis, :] - self.remain_positions[np.newaxis, :, :]
                dist_matrix = np.linalg.norm(diff, axis=-1)
                np.fill_diagonal(dist_matrix, np.inf)
                total_collisions += np.sum(dist_matrix < 10.0) // 2

            _, num = self.check_number_of_clusters()
            if num == 1:
                connected_flag = True
                break

        if trajectory_log:
            last_entry = trajectory_log[-1]
            trajectory_log.append(
                {
                    "positions": deepcopy(self.true_positions).astype(np.float32),
                    "velocities": np.zeros_like(last_entry["velocities"]),
                    "remain_mask": last_entry["remain_mask"],
                    "damaged_mask": last_entry["damaged_mask"],
                }
            )

        step_collisions = total_collisions / (step + 1)
        return step, self.remain_positions, num, connected_flag, trajectory_log, step_collisions


if __name__ == "__main__":
    np.random.seed(57)
    environment = Environment()
    swarm = Swarm(algorithm_mode="PhyGAIL")

    environment_positions = environment.reset()
    _, destroy_list = environment.stochastic_destroy(mode=2, num_of_destroyed=10)
    swarm.destroy_happens(deepcopy(destroy_list), deepcopy(environment_positions))
    print(swarm.solve()[:4])
