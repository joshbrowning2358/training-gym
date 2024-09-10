import gymnasium as gym

gym.register(id="DollarGame-v0", entry_point="dollar_game.dollar_game:DollarGame")
dollar_game = gym.make("DollarGame-v0")

gym.register(id="farkle", entry_point="farkle.farkle:Farkle")
farkle = gym.make("farkle")

gym.register(id="FrozenLakeN-v0", entry_point="frozen_lake.frozen_lake_n:FrozenLakeN")
frozen_lake_n = gym.make("FrozenLakeN-v0", render_mode="human")

gym.register(id="FrozenLakeCustom-v0", entry_point="frozen_lake.frozen_lake_custom:FrozenLakeCustom")
frozen_lake_custom = gym.make("FrozenLakeCustom-v0", render_mode="human")

gym.register(id="NumberGuessing-v0", entry_point="number_guessing.env:NumberGuessing")
number_guessing = gym.make("NumberGuessing-v0")
