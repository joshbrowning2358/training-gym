import json
import os
from typing import List

import click
import gymnasium as gym
from stable_baselines3 import PPO

from rl.utils import make_model_path


@click.command("infer-rl")
@click.option("--experiment-name", type=str)
@click.option("--epochs", type=int, multiple=True)
@click.option("--output-path", type=click.Path(), default="/home/josh/output")
def infer_rl(experiment_name: str, epochs: List[int], output_path: str) -> None:
    with open(os.path.join(output_path, experiment_name, "info.json"), "r") as f:
        info = json.load(f)

    env = gym.make(info["env_name"], render_mode="human")
    for epoch in epochs:
        model = PPO.load(make_model_path(output_path, experiment_name, epoch))

        obs, _ = env.reset()
        done = False
        while not done:
            action, states = model.predict(obs)
            try:
                obs, rewards, bool1, bool2, info = env.step(int(action))
            except:
                obs, rewards, bool1, bool2, info = env.step(action)
            done = bool1 or bool2
            env.render()


if __name__ == "__main__":
    infer_rl()
