import importlib
import sys
from logging import warning
from time import sleep

import click
import gymnasium as gym
import numpy as np

# "CliffWalking-v0"
#   Observation is current_row * nrows + current_column
#   Action is 0 (north), 1 (west), 2 (south), 3 (east)
# "Taxi-v3":
#   Observation is ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
#   Action is 0 (south), 1 (north), 2 (east), 3 (west), 4 (pickup), 5 (dropoff)
#   0: Red, 1: Green, 2: Yellow, 3: Blue
# "FrozenLake-v1", "FrozenLake8x8-v1"
#   Observation is current_row * nrows + current_column
#   Action is 0 (west), 1 (south), 2 (east), 3 (north)
# "Blackjack-v1"
#   Observation is (player_sum, dealer_card, usable_ace)
#   Action is 0 (stick), 1 (hit)
# "MountainCar-v0"
#   Observation is (position, velocity)
#   Action is 0 (push left), 1 (no push), 2 (push right)
# "ALE/Breakout-v5"
#   Observation is (210, 160,s 3) array of uint8 (RGB screen of the game)
#   Action is 0 (noop), 1 (fire), 2 (right), 3 (left)


@click.command("run-game")
@click.option("--game-name", type=str)
@click.option("--function-name", type=str)
@click.option("--script-location", type=str, default="agents")
@click.option("--n-games", type=int, default=1)
def run_game(game_name: str, function_name: str, script_location: str, n_games: int):
    module = importlib.import_module(script_location)
    get_action = getattr(module, function_name)

    env = gym.make(game_name, render_mode="human")
    print(f"Action space is {env.action_space}")

    observation, info = env.reset()
    print("Observation: ", observation)
    game_rewards = []
    for i in range(n_games):
        terminated = False
        truncated = False
        total_reward = 0
        while (not terminated) and (not truncated):
            action = 0
            try:
                action = get_action(observation, info)
            except BaseException:
                warning("get_action() call failed, using 0 as action!")
            if action not in env.action_space:
                warning(f"Action {action} not in valid actions {env.action_space}, replacing with 0!")
                action = 0

            observation, reward, terminated, truncated, info = env.step(action)
            print("Observation: ", observation)

            total_reward += reward

            env.render()
            if env.spec.id == "CartPole-v1":
                sleep(0.1)

        print(f"Game over, score was {total_reward}!")
        game_rewards += [total_reward]
        observation, info = env.reset()

    env.close()

    print(f"Rewards were {game_rewards}, average reward was {np.mean(game_rewards)}")


if __name__ == "__main__":
    run_game(sys.argv[1:])
