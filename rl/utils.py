import os


def make_model_path(output_path: str, experiment_name: str, epoch: int) -> str:
    return os.path.join(output_path, experiment_name, f"epoch_{epoch}")
