from logging import warning

import gymnasium as gym
import numpy as np
import tensorflow as tf

from deep_q_network import ConvModel, preprocess

model = ConvModel(channels=[16, 128], kernel_sizes=[4, 8], strides=[4, 8], n_actions=6)
model.build(input_shape=(1, 169, 128, 4))
model.load_weights("/home/josh/output/galaxian/first_attempt/epoch_0")

# env = gym.make("ALE/Galaxian-v5", render_mode="rgb_array")
env = gym.make("ALE/Galaxian-v5", render_mode="human")


observation, info = env.reset(seed=42)
games_to_play = 5
game_rewards = []
for i in range(games_to_play):
    terminated = False
    truncated = False
    total_reward = 0
    image_queue = [preprocess(observation)]
    while (not terminated) and (not truncated):
        action = 0
        if len(image_queue) == 4:
            tf_input = tf.cast(tf.stack(image_queue, axis=2), tf.float32)
            tf_input = tf.expand_dims(tf_input, axis=0)  # Add batch dim
            rewards = model.predict(tf_input)
            action = np.argmax(rewards)
        else:
            action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)
        image_queue += [preprocess(observation)]
        if len(image_queue) > 4:
            image_queue.pop(0)

        total_reward += reward

        env.render()

    print(f"Game over, score was {total_reward}!")
    game_rewards += [total_reward]
    observation, info = env.reset()

env.close()

print(f"Rewards were {game_rewards}")
