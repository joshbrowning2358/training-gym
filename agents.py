from logging import warning

import numpy as np


def get_action(observation, info):
    return (observation[1] > 0) * 2


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
