from logging import warning

import numpy as np


def get_action(observation, info):
    return (observation["high"] + observation["low"]) // 2


def get_mountain_car_right_action(observation, info):
    return 2


def get_action_random4(observation, info):
    return np.random.choice([0, 1, 2, 3])


def get_action_random2(observation, info):
    return np.random.choice([0, 1])


def get_interactive_action(observation, info) -> int:
    got_action = False
    while not got_action:
        action_str = input("Enter action: ")
        try:
            action = int(action_str)
            got_action = True
        except:  # noqa
            warning(f"Invalid action {action_str}!")
    return action


def get_interactive_farkle_action(observation, info) -> dict:
    got_action = False
    while not got_action:
        action_str = input("Enter action (0/1s separated by commas): ")
        try:
            action = tuple(map(int, action_str.split(",")))
            got_action = True
            if len(action) != 7:
                warning("Invalid action length!  Except 6 keep/no keeps and 1 stop/continue.")
                got_action = False
        except:  # noqa
            warning(f"Invalid action {action_str}!")
    return {"keep": tuple(action[:6]), "stop": action[6]}
