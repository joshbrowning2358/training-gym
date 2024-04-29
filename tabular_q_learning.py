import importlib
from collections import defaultdict

import click
import gymnasium as gym
import numpy as np
from tqdm import tqdm


@click.command("fit-tablular-q-learner")
@click.option("--env-name", type=str, help="Environment to fit the model on", required=True)
@click.option("--learning-rate", type=float, help="Learning rate", default=0.01)
@click.option("--custom-game", type=bool, is_flag=True)
@click.option("--num-episodes", type=int, help="Number of episodes to train for", default=1000)
@click.option("--num-evals", type=int, help="How many evaluations to run throughout the entire training", default=10)
def fit_model(
    env_name: str, learning_rate: int, custom_game: bool, num_episodes: int, num_evals: int
) -> defaultdict[int, dict[int, float]]:
    """
    Fit a tabular Q learner model on the given environment.
    """
    if num_episodes % num_evals != 0:
        raise ValueError("num_episodes must be divisible by num_evals")

    model = TabularQLearner(env_name, learning_rate=learning_rate, custom_game=custom_game)
    for _ in tqdm(range(num_evals)):
        model.fit(num_episodes // num_evals)
        model.run()
    return model.q_table


class TabularQLearner:
    def __init__(self, env_name, learning_rate: int, custom_game: bool):
        self.env_name = env_name
        self.custom_game = custom_game
        self.learning_rate = learning_rate
        env = self._make_env(None)
        self.q_table: defaultdict[int, dict[int, float]] = defaultdict(
            lambda: {a: 0 for a in range(env.action_space.n)}
        )
        self.discount_factor = 0.9  # Discount factor
        self.action_space = env.action_space

    def _make_env(self, render_mode):
        if self.custom_game:
            module = importlib.import_module("custom_games")
            return getattr(module, self.env_name)
        else:
            return gym.make(self.env_name, render_mode=render_mode)

    def fit(self, num_episodes: int) -> None:
        env = self._make_env(render_mode=None)
        for i in range(num_episodes):
            state, _ = env.reset()
            done = False
            while not done:
                action = self.choose_action(state, epsilon=0.1)
                next_state, reward, done, _, _ = env.step(action)
                self.q_table[state][action] = (1 - self.learning_rate) * self.q_table[state][
                    action
                ] + self.learning_rate * (reward + self.discount_factor * max(self.q_table[next_state].values()))
                state = next_state

    def choose_action(self, state, epsilon: float) -> int:
        if np.random.uniform() < epsilon:
            return self.action_space.sample()
        best_value = max(self.q_table[state].values())
        return np.random.choice([a for a, v in self.q_table[state].items() if v == best_value])

    def run(self):
        env = self._make_env(render_mode="human")
        state, _ = env.reset()
        done = False
        while not done:
            action = self.choose_action(state, epsilon=0)
            state, _, done, _, _ = env.step(action)
            env.render()
        env.close()


if __name__ == "__main__":
    fit_model()
