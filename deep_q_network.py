import sys

import click
import gymnasium as gym
import numpy as np
import tensorflow as tf
from PIL import Image
from tf_agents.replay_buffers import TFUniformReplayBuffer

from constants import PLAY_COLS, PLAY_ROWS

N_TRAINING_IMAGES = 4


def produce_training_data(env, model):
    env.reset()
    # replay_buffer = get_replay_buffer()
    image_queue = []
    while len(image_queue) < N_TRAINING_IMAGES:
        observation, _, _, _, _ = env.step(env.action_space.sample())
        image_queue += [preprocess(observation)]
    # observation, reward, terminated, truncated, info = env.step(action)


def preprocess(img: np.ndarray) -> np.ndarray:
    img = img[PLAY_ROWS[0] : PLAY_ROWS[1], PLAY_COLS[0] : PLAY_COLS[1]]
    img = np.array(Image.fromarray(img).convert("L"))
    return img


def get_replay_buffer():
    # Crop play image to rows 16:185, cols 16:144 -> 169x128
    data_spec = (
        tf.TensorSpec(shape=(1,), dtype=tf.uint8, name="action"),
        tf.TensorSpec(shape=(169, 128, N_TRAINING_IMAGES), dtype=tf.uint8, name="4x_bw_game_images"),
    )

    batch_size = 32
    max_length = 1000

    replay_buffer = TFUniformReplayBuffer(data_spec, batch_size=batch_size, max_length=max_length)
    return replay_buffer


@click.command("run-training")
@click.option("environment-name", type=str, default="ALE/Galaxian-v5")
def run_training(environment_name: str):
    produce_training_data(env=gym.make(environment_name, render_mode="rgb_array"), model=None)


if __name__ == "__main__":
    run_training(sys.argv[1:])
