import gymnasium as gym

map = []
N = 20
for i in range(1, N - 1):
    pre_holes = max(i - 2, 0)
    post_holes = max(N - i - 3, 0)
    ice = N - pre_holes - post_holes
    map += ["H" * pre_holes + "F" * ice + "H" * post_holes]

map = ["S" + 2 * "F" + (N - 3) * "H"] + map + ["H" * (N - 3) + "F" * 2 + "G"]

frozen_lake = gym.make("FrozenLake-v1", desc=map, is_slippery=False, render_mode="human")


gym.register(
    id="DollarGame-v0",
    entry_point="dollar_game.dollar_game:DollarGame",
)

dollar_game = gym.make("DollarGame-v0")


gym.register(
    id="NumberGuessing-v0",
    entry_point="number_guessing.env:NumberGuessing",
)

number_guessing = gym.make("NumberGuessing-v0")
