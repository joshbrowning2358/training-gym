from logging import warning
from typing import Dict, List

import numpy as np
from scipy.signal import convolve2d

from constants import LEFT_RIGHT_BORDER, PLAY_COLS

ENEMY_ROWS = (17, 90)
APPROACHING_ROWS = (90, 185)
DEFENDER_ROWS = (166, 185)

ACTION_SHOOT = 1
ACTION_RIGHT_SHOOT = 4
ACTION_LEFT_SHOOT = 5


def get_action_random(observation, info: Dict):
    return np.random.choice([ACTION_SHOOT, ACTION_RIGHT_SHOOT, ACTION_LEFT_SHOOT])


def get_pole_cart_random_action(observation, info: Dict):
    return np.random.choice([0, 1])


def get_mountain_car_right_action(observation, info: Dict):
    return 2


def get_interactive_action(observation, info: Dict) -> int:
    got_action = False
    while not got_action:
        action_str = input("Enter action: ")
        try:
            action = int(action_str)
            got_action = True
        except:  # noqa
            warning(f"Invalid action {action_str}!")
    return action


def get_pole_cart_interactive_action(observation, info: Dict) -> int:
    got_action = False
    while not got_action:
        action_str = input("Enter action: ")
        try:
            action = int(action_str)
            if action not in [0, 1]:
                raise ValueError("Invalid action, should be 0 or 1!")
        except:  # noqa
            warning(f"Invalid action {action_str}, should be 0 or 1!")
        got_action = True
    return action


def get_mountain_car_interactive_action(observation, info: Dict) -> int:
    got_action = False
    while not got_action:
        action_str = input("Enter action: ")
        try:
            action = int(action_str)
            if action not in [0, 1, 2]:
                raise ValueError("Invalid action, should be 0, 1 or 2!")
        except:  # noqa
            warning(f"Invalid action {action_str}, should be 0, 1 or 2!")
        got_action = True
    return action


def get_action_avoid_alien(observation, info: Dict):

    # from PIL import Image
    # player = Image.fromarray(observation[166:185, 76:86])
    # player_row = Image.fromarray(observation[166:185, 16:144])
    # enemy_rows = Image.fromarray(observation[17:90, 16:144])

    template = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 210, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 210, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 210, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 236, 236, 0, 0, 0, 0],
            [0, 0, 0, 236, 236, 236, 236, 0, 0, 0],
            [0, 0, 236, 0, 236, 236, 0, 236, 0, 0],
            [0, 0, 0, 0, 236, 236, 0, 0, 0, 0],
            [0, 0, 0, 236, 236, 236, 236, 0, 0, 0],
            [0, 0, 0, 0, 236, 236, 0, 0, 0, 0],
            [0, 236, 0, 0, 236, 236, 0, 0, 236, 0],
            [0, 236, 0, 236, 236, 236, 236, 0, 236, 0],
            [0, 236, 236, 236, 236, 236, 236, 236, 236, 0],
            [0, 236, 236, 236, 236, 236, 236, 236, 236, 0],
            [0, 236, 236, 0, 236, 236, 0, 236, 236, 0],
            [0, 236, 0, 0, 236, 236, 0, 0, 236, 0],
            [0, 236, 0, 0, 0, 0, 0, 0, 236, 0],
            [0, 236, 0, 0, 0, 0, 0, 0, 236, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    player_location = convolve2d(
        observation[DEFENDER_ROWS[0] : DEFENDER_ROWS[1], PLAY_COLS[0] : PLAY_COLS[1], 0], template, mode="valid"
    )
    player_location = player_location.argmax() + template.shape[1] // 2 + LEFT_RIGHT_BORDER

    # We've identified our location, now zero those pixels so we can identify enemies on the whole screen
    observation[
        DEFENDER_ROWS[0] : DEFENDER_ROWS[1],
        player_location - template.shape[1] // 2 : player_location + template.shape[1] // 2,
        :,
    ] = 0
    # Remover defender shots based on color (don't want to run away from your own shot)!
    mask = observation[APPROACHING_ROWS[0] : APPROACHING_ROWS[1], PLAY_COLS[0] : PLAY_COLS[1]] == [210, 164, 74]
    observation[APPROACHING_ROWS[0] : APPROACHING_ROWS[1], PLAY_COLS[0] : PLAY_COLS[1]][mask] = 0

    approaching_enemy = observation[APPROACHING_ROWS[0] : APPROACHING_ROWS[1], PLAY_COLS[0] : PLAY_COLS[1]]
    if approaching_enemy.max() > 0:
        return avoid_enemy_flee(observation, player_location)
    else:
        # "Safe": decide how to move next
        # return safe_action_middle(observation, player_location)
        return safe_action_target_enemy(observation, player_location)


def safe_action_middle(img: np.ndarray, player_location: int) -> int:
    if player_location < 80:
        return ACTION_RIGHT_SHOOT
    else:
        return ACTION_LEFT_SHOOT


def safe_action_target_enemy(img: np.ndarray, player_location: int) -> int:
    enemy_rows = img[ENEMY_ROWS[0] : ENEMY_ROWS[1], PLAY_COLS[0] : PLAY_COLS[1]]
    enemy_rows = enemy_rows.sum(axis=(0, 2))

    columns: List[int] = []
    current_left = None
    for i, val in enumerate(enemy_rows):
        if val == 0:
            if current_left is not None:
                columns += [(current_left + i - 1) // 2 + LEFT_RIGHT_BORDER]
            current_left = None
        elif val > 0:
            current_left = i if current_left is None else current_left

    if len(columns) == 0:
        warning("No enemies found, returning 1 to just shoot!")
        return ACTION_SHOOT

    rel_dist = np.abs(player_location - np.array(columns))
    target_column = columns[rel_dist.argmin()]
    if player_location < target_column:
        return ACTION_RIGHT_SHOOT
    elif player_location > target_column:
        return ACTION_LEFT_SHOOT
    else:
        # Right where we want to be, shoot!
        return ACTION_SHOOT


def avoid_enemy_flee(img: np.ndarray, player_location: int, safe_distance: int = 25) -> int:
    enemy_location = (
        img[APPROACHING_ROWS[0] : APPROACHING_ROWS[1], PLAY_COLS[0] : PLAY_COLS[1]].sum(axis=(0, 2)).argmax()
        + LEFT_RIGHT_BORDER
    )

    if abs(enemy_location - player_location) > safe_distance:
        # Far away, take a safe action:
        return safe_action_target_enemy(img, player_location)

    if enemy_location > player_location:
        return ACTION_LEFT_SHOOT
    else:
        return ACTION_RIGHT_SHOOT


def get_action(observation, info):
    row, col = get_coordinate(observation)
    return (row > col) + 1


def get_coordinate(observation, N=20):
    return observation // N, observation % N


def get_action_dollar(observation, info):
    if observation["draw"] < 86:
        return 1
    else:
        return 0


def get_action_number_guessing(observation, info):
    return (observation["low"] + observation["high"]) // 2
