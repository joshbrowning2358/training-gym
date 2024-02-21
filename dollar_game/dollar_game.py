from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import Env


class DollarGame(Env):
    def __init__(self):
        super(DollarGame, self).__init__()

        space = {
            "draw": gym.spaces.Discrete(start=1, n=100),
            "times_paid": gym.spaces.Discrete(n=1000),
        }
        self.observation_space = gym.spaces.Dict(space)

        self.action_space = gym.spaces.Discrete(n=2)
        self.times_paid = 0
        self.draw = None

    def reset(self) -> Tuple[Dict[str, int], Dict]:
        self.times_paid = 0
        self.draw = np.random.choice(range(1, 101))
        return {"draw": self.draw, "times_paid": self.times_paid}, {}

    def render(self, mode="human"):
        print(f"You drew ${self.draw}!  What would you like to do?  0 to keep, 1 to pay $1 and play again")

    @staticmethod
    def get_action_meanings():
        return {0: "keep", 1: "pay $1 and play again"}

    def step(self, action) -> Tuple[Dict[str, int], int, bool, bool, Dict]:
        """
        Returns:
            Tuple of observation, reward, terminated, truncated, info
        """
        if action == 0:
            winnings = self.draw - self.times_paid
            return {"draw": self.draw, "times_paid": self.times_paid}, winnings, True, False, {}
        elif action == 1:
            self.times_paid += 1
            self.draw = np.random.choice(range(1, 101))
            return {"draw": self.draw, "times_paid": self.times_paid}, 0, False, False, {}
        else:
            raise ValueError(f"Invalid action {action}!")
