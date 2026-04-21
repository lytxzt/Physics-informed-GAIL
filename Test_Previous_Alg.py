from copy import deepcopy
import argparse

from Configurations import *
from Environment import Environment
from Swarm import PREVIOUS_ALGORITHM_ORDER, Swarm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        type=str,
        default="centering",
        choices=PREVIOUS_ALGORITHM_ORDER,
        help="Previous baseline to test.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    environment = Environment()
    swarm = Swarm(algorithm_mode=args.algorithm)

    environment_positions = environment.reset()
    swarm.reset()

    destroy_num = min(config_num_destructed_UAVs, config_num_of_agents // 2)
    _, destroy_list = environment.stochastic_destroy(mode=2, num_of_destroyed=destroy_num)
    swarm.destroy_happens(deepcopy(destroy_list), deepcopy(environment_positions))

    step, _, num_subnets, connected_flag, _, step_collisions = swarm.solve()
    print("=" * 60)
    print("Test_Previous_Alg")
    print(f"Algorithm: {args.algorithm}")
    print(f"Agents: {config_num_of_agents}")
    print(f"Destroyed: {destroy_num}")
    print(f"Connected: {connected_flag}")
    print(f"Recovery steps: {step + 1}")
    print(f"Final subnet count: {num_subnets}")
    print(f"Average collisions per step: {step_collisions:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
