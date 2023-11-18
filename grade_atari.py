from logging import warning

import gymnasium as gym

# from example import get_action_zero as get_action
from agents import get_action_avoid_alien as get_action

# from agents import get_action_random as get_action

# env = gym.make("ALE/Galaxian-v5", render_mode="rgb_array")
env = gym.make("ALE/Galaxian-v5", render_mode="human")


observation, info = env.reset(seed=42)
games_to_play = 5
game_rewards = []
for i in range(games_to_play):
    terminated = False
    truncated = False
    total_reward = 0
    while (not terminated) and (not truncated):
        action = 0
        try:
            action = get_action(observation, info["lives"], info["frame_number"])
        except BaseException:
            warning("get_action() call failed, using 0 as action!")
        if action not in env.action_space:
            warning(f"Action {action} not in valid actions {env.action_space}, replacing with 0!")
            action = 0

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        env.render()

    print(f"Game over, score was {total_reward}!")
    game_rewards += [total_reward]
    observation, info = env.reset()

env.close()

print(f"Rewards were {game_rewards}")
