import gymnasium as gym
import numpy as np
import tensorflow as tf

from training_gym.deep_q_network import preprocess

# model = tf.saved_model.load("/home/josh/output/galaxian/first_attempt/epoch_2")
# model = tf.saved_model.load("/home/josh/output/galaxian/2023-11-22_death_penalty/epoch_3")
# model = tf.saved_model.load("/home/josh/output/galaxian/2023-11-23_batch_size16/epoch_11")
# model = tf.saved_model.load("/home/josh/output/galaxian/2023-11-28_batch_size32_ram_tricks/epoch_4")
# model = tf.saved_model.load("/home/josh/output/galaxian/2024-04-29/epoch_19")
model = tf.saved_model.load("/home/josh/output/galaxian/2024-04-30_bs64_dis999_ep100/epoch_28")

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
            # tf_input = tf.cast(tf.stack(image_queue, axis=2), tf.float32)
            # tf_input = tf.expand_dims(tf_input, axis=0)  # Add batch dim
            tf_input = np.expand_dims(np.stack(image_queue, axis=2), axis=0)
            rewards = model.serve(tf_input)

            model.serve(np.expand_dims(np.stack(image_queue, axis=2), axis=0))

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
