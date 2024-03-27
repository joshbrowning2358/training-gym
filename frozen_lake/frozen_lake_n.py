from typing import Optional

import numpy as np
from gymnasium.envs.toy_text import FrozenLakeEnv


class FrozenLakeN(FrozenLakeEnv):
    def __init__(self, **kwargs):
        map = []
        self.board_size = np.random.randint(8, 40)
        for i in range(1, self.board_size - 1):
            pre_holes = max(i - 2, 0)
            post_holes = max(self.board_size - i - 3, 0)
            ice = self.board_size - pre_holes - post_holes
            map += ["H" * pre_holes + "F" * ice + "H" * post_holes]
        map = ["S" + 2 * "F" + (self.board_size - 3) * "H"] + map + ["H" * (self.board_size - 3) + "F" * 2 + "G"]
        super(FrozenLakeN, self).__init__(desc=map, is_slippery=False, **kwargs)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        state, info = super(FrozenLakeN, self).reset()
        state = {"position": state, "board_size": self.board_size}
        return state, info

    def step(self, a):
        state, x1, x2, x3, x4 = super(FrozenLakeN, self).step(a)
        state = {"position": state, "board_size": self.board_size}
        return state, x1, x2, x3, x4
