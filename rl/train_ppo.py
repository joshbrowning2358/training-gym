import json
import os

import click
import gymnasium as gym
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3 import PPO

from rl.utils import make_model_path


@click.command("train-rl")
@click.option("--steps-per-epoch", type=int)
@click.option("--num-epochs", type=int)
@click.option("--experiment-name", type=str)
@click.option("--env-name", type=str, default="ALE/Galaxian-v5")
@click.option("--output-path", type=click.Path(), default="/home/josh/output")
@click.option("--policy-name", type=click.Choice(["ActorCriticCnnPolicy", "MlpPolicy"]))
def train_rl(
    steps_per_epoch: int, num_epochs: int, experiment_name: str, env_name: str, output_path: str, policy_name: str
) -> None:
    env = gym.make(env_name)

    save_path = os.path.join(output_path, experiment_name)
    if os.path.exists(save_path):
        epochs = [x for x in os.listdir(save_path) if x.startswith("epoch")]
        start_epoch = max(int(x.replace("epoch_", "").replace(".zip", "")) for x in epochs) + 1
        model = PPO.load(os.path.join(save_path, f"epoch_{start_epoch - 1}.zip"), env=env, verbose=1)
    else:
        start_epoch = 0
        os.makedirs(os.path.join(output_path, experiment_name), exist_ok=True)
        with open(os.path.join(output_path, experiment_name, "info.json"), "w") as f:
            json.dump(obj={"env_name": env_name}, fp=f, indent=4)

        if policy_name == "ActorCriticCnnPolicy":
            model = PPO(policy=ActorCriticCnnPolicy, env=env, verbose=1)
        else:
            model = PPO(policy=MlpPolicy, env=env, verbose=1)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.learn(total_timesteps=steps_per_epoch)
        model.save(make_model_path(output_path, experiment_name, epoch))


if __name__ == "__main__":
    train_rl()
