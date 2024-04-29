from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import Env


class NumberGuessing(Env):
    def __init__(self, num_values: int = 1000000000):
        super(NumberGuessing, self).__init__()

        self.num_values = num_values

        self.action_space = gym.spaces.Discrete(n=num_values + 1)
        self.observation_space = gym.spaces.Dict(
            {"high": gym.spaces.Discrete(n=num_values + 1), "low": gym.spaces.Discrete(n=num_values)}
        )
        self.high = self.num_values
        self.low = 1
        self.num_guesses = 0
        self._answer = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, int], Dict]:
        self._answer = np.random.choice(range(1, self.num_values + 1))
        self.high = self.num_values
        self.low = 1
        self.num_guesses = 0
        return {"high": self.high, "low": self.low}, {}

    def render(self, mode="human"):
        print(f"Guess a number between 1 and {self.num_values}, current range is {self.low} to " f"{self.high}!")

    def step(self, action):
        """
        Returns:
            Tuple of observation, reward, terminated, truncated, info
        """
        self.num_guesses += 1
        if action == self._answer:
            return {"high": self._answer, "low": self._answer}, self.num_guesses, True, False, {}
        else:
            if action > self._answer:
                self.high = min(self.high, action - 1)
                print("Guess was too high!")
            else:
                self.low = max(self.low, action + 1)
                print("Guess was too low!")
            return {"high": self.high, "low": self.low}, 0, False, False, {}
