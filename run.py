import gym
import numpy as np

#env = gym.make("ALE/Galaxian-v5", render_mode="rgb_array")
env = gym.make("ALE/Galaxian-v5", render_mode="human")

def get_action(observation, lives, frame_number):
    return np.random.choice([0, 1, 2, 3, 4, 5])


observation, info = env.reset(seed=42)
games_to_play = 5
game_rewards = []
for i in range(games_to_play):
    terminated = False
    truncated = False
    total_reward = 0
    while (not terminated) and (not truncated):
        # action = env.action_space.sample()
        action = get_action(observation, info["lives"], info["frame_number"])
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        env.render()

    print(f"Game over, score was {total_reward}!")
    game_rewards += [total_reward]
    observation, info = env.reset()

env.close()

print(f"Rewards were {game_rewards}")
