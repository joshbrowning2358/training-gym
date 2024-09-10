from typing import Optional

from gymnasium.envs.toy_text import FrozenLakeEnv

MAP = [
    ["S", "F", "F"],
    ["F", "F", "F"],
    ["F", "F", "F"],
    ["F", "F", "G"],
]


class FrozenLakeCustom(FrozenLakeEnv):
    def __init__(self, **kwargs):
        map = []
        self.board_size = (len(MAP), len(MAP[0]))
        for i in range(1, self.board_size[0] - 1):
            pre_holes = max(i - 2, 0)
            post_holes = max(self.board_size[0] - i - 3, 0)
            ice = self.board_size[0] - pre_holes - post_holes
            map += ["H" * pre_holes + "F" * ice + "H" * post_holes]
        map = MAP
        super(FrozenLakeCustom, self).__init__(desc=map, is_slippery=False, **kwargs)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        state, info = super(FrozenLakeCustom, self).reset()
        state = {"position": state, "board_size": self.board_size}
        return state, info

    def step(self, a):
        state, x1, x2, x3, x4 = super(FrozenLakeCustom, self).step(a)
        state = {"position": state, "board_size": self.board_size}
        return state, x1, x2, x3, x4
