import gc
import os
import sys
from datetime import datetime
from typing import List

import click
import gymnasium as gym
import keras
import numpy as np
import tensorflow as tf
from keras.src.callbacks import ModelCheckpoint
from PIL import Image
from tf_agents.replay_buffers import TFUniformReplayBuffer
from tqdm import tqdm

from constants import PLAY_COLS, PLAY_ROWS

N_TRAINING_IMAGES = 4  # Number of images in history to train on
BATCH_SIZE = 32
MAX_LENGTH = 100  # Max replay buffer size, will have this many obs * BATCH_SIZE
NUM_STEPS = 1000  # Train steps per training epoch
DISCOUNT_FACTOR = 0.999
NUM_EPOCHS = 20
DEATH_PENALTY = -500  # Reward added when life is lost.  Maybe should be ~typical score for one life?


def produce_training_data(env, model, replay_buffer=None):
    env.reset()
    if replay_buffer is None:
        replay_buffer = get_replay_buffer()
    image_queue = []
    while len(image_queue) < N_TRAINING_IMAGES:
        observation, _, _, _, _ = env.step(env.action_space.sample())
        image_queue += [preprocess(observation)]

    current_lives = 3
    batch_collector = {"imgs": [], "rewards": [], "actions": [], "next_obs": []}
    for _ in tqdm(range(MAX_LENGTH)):
        added = 0
        while added < BATCH_SIZE:
            imgs = np.expand_dims(np.stack(image_queue, axis=2), axis=0).astype(np.float32)
            action_rewards = model.predict(imgs, verbose=0)
            action = np.argmax(action_rewards)
            # Add randomness to actions to increase variability
            if np.random.random() < 0.1:
                action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated:
                env.reset(seed=int(datetime.now().timestamp()))

            if info["lives"] < current_lives or terminated:
                reward += DEATH_PENALTY
                current_lives = info["lives"]

            # Accumulate experience
            batch_collector["imgs"] += [imgs]
            batch_collector["rewards"] += [reward]
            batch_collector["actions"] += [action]
            batch_collector["next_obs"] += [preprocess(observation)]
            added += 1

            # Write to replay buffer
            if len(batch_collector["imgs"]) == replay_buffer._batch_size:
                replay_buffer.add_batch(
                    (
                        tf.reshape(tf.stack(batch_collector["actions"]), (-1, 1)),
                        tf.reshape(tf.stack(batch_collector["rewards"]), (-1, 1)),
                        tf.cast(tf.concat(batch_collector["imgs"], axis=0), tf.uint8),
                        tf.cast(batch_collector["next_obs"], tf.uint8),
                    )
                )
                batch_collector = {"imgs": [], "rewards": [], "actions": [], "next_obs": []}

            image_queue.pop(0)
            image_queue.append(preprocess(observation))

    return replay_buffer


def preprocess(img: np.ndarray) -> np.ndarray:
    img = img[PLAY_ROWS[0] : PLAY_ROWS[1], PLAY_COLS[0] : PLAY_COLS[1]]
    img = np.array(Image.fromarray(img).convert("L"))
    return img


def get_replay_buffer():
    # Crop play image to rows 16:185, cols 16:144 -> 169x128
    data_spec = (
        tf.TensorSpec(shape=(1,), dtype=tf.int32, name="action"),
        tf.TensorSpec(shape=(1,), dtype=tf.float32, name="reward"),
        tf.TensorSpec(shape=(169, 128, N_TRAINING_IMAGES), dtype=tf.uint8, name="4x_bw_game_images"),
        tf.TensorSpec(shape=(169, 128), dtype=tf.uint8, name="next_game_image"),
    )

    replay_buffer = TFUniformReplayBuffer(data_spec, batch_size=BATCH_SIZE, max_length=MAX_LENGTH)
    return replay_buffer


def train_model(model, replay_buffer, checkpoint_callback):
    dataset = replay_buffer.as_dataset(sample_batch_size=BATCH_SIZE)
    iterator = iter(dataset)

    for step in tqdm(range(NUM_STEPS), desc="training", unit="steps"):
        (action, reward, imgs, next_img), _ = next(iterator)
        # action, reward, imgs, next_img = replay_buffer.get_next(sample_batch_size=BATCH_SIZE)[0]

        # We train our model by forcing the current value = discounted future value + reward.  We have the reward, but
        # the discounted future value requires a model inference pass.  We need 4 images to run inference, so we should
        # use the last 3 in the training dataset + next one.
        next_obs = tf.concat([imgs[:, :, :, 1:], tf.expand_dims(next_img, axis=3)], axis=3)
        next_obs = tf.cast(next_obs, tf.float32)
        estimated_future_value = model.predict(next_obs, verbose=0)
        # estimated_future_value is a function of action, so take max
        estimated_future_value = tf.reduce_max(estimated_future_value, axis=1)
        target = estimated_future_value * DISCOUNT_FACTOR + reward[:, 0]

        if step == NUM_STEPS - 1:
            model.fit(tf.cast(imgs, tf.float32), target, verbose=0, callbacks=[checkpoint_callback])
        else:
            model.fit(tf.cast(imgs, tf.float32), target, verbose=0)

    return model


class ConvModel(tf.keras.Model):
    def __init__(self, channels: List[int], kernel_sizes: List[int], strides: List[int], n_actions: int):
        super().__init__()
        self.conv_layers = []
        for channel, kernel_size, stride in zip(channels, kernel_sizes, strides):
            self.conv_layers += [keras.layers.Conv2D(filters=channel, kernel_size=kernel_size, strides=stride)]
        self.flatten = tf.keras.layers.Flatten()
        self.dense_layer = keras.layers.Dense(units=n_actions, activation="softmax")

    def call(self, imgs):
        x = imgs
        for conv in self.conv_layers:
            x = conv(x)
        x = self.flatten(x)
        x = self.dense_layer(x)
        return x


@click.command("run-training")
@click.option("--environment-name", type=str, default="ALE/Galaxian-v5")
@click.option("--save-dir", type=click.Path())
def run_training(environment_name: str, save_dir: str):
    # if os.path.exists(save_dir):
    #     model = tf.saved_model.load()
    os.makedirs(save_dir, exist_ok=True)

    env = gym.make(environment_name, render_mode="rgb_array")
    model = ConvModel(channels=[16, 128], kernel_sizes=[4, 8], strides=[4, 8], n_actions=env.action_space.n)
    checkpoint_callback = ModelCheckpoint(filepath=save_dir)
    model.build(input_shape=(1, 169, 128, 4))
    model.compile(optimizer="Adam", loss="mse")
    replay_buffer = None
    for epoch in range(NUM_EPOCHS):
        replay_buffer = produce_training_data(env, model=model, replay_buffer=replay_buffer)
        train_model(model, replay_buffer, checkpoint_callback)
        model.export(os.path.join(save_dir, f"epoch_{epoch}"))
        replay_buffer.clear()
        del replay_buffer
        gc.collect()


if __name__ == "__main__":
    run_training(sys.argv[1:])
