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


SOUTH = 0
NORTH = 1
EAST = 2
WEST = 3
PICKUP = 4
DROPOFF = 5

RED = (0, 0)
GREEN = (0, 4)
YELLOW = (4, 0)
BLUE = (4, 3)


# Taxi cab
def get_action(observation, info: Dict):
    destination, passenger, taxi_row, taxi_col = extract_observation(observation)
    print(f"Destination: {destination}, Passenger: {passenger}, Taxi: ({taxi_row}, {taxi_col})")
    dest = destination if passenger == "taxi" else passenger
    # At destination, need to pick up or drop off passenger
    if (taxi_row, taxi_col) == dest:
        return DROPOFF if passenger == "taxi" else PICKUP

    # Not at destination, need to move to it
    # If on row 2, we can move east/west freely
    if taxi_row == 2:
        if taxi_col < dest[1]:
            return EAST
        elif taxi_col > dest[1]:
            return WEST
        else:
            return SOUTH if taxi_row < dest[0] else NORTH

    # If vertically far away from destination, move to row 2
    if abs(taxi_row - dest[0]) > 2:
        return SOUTH if taxi_row < 2 else NORTH

    if taxi_row < 2:
        # If vertically close to destination but blocked by col, move to row 2
        if (taxi_col < 2 and dest[1] == 4) or (taxi_col >= 2 and dest[1] == 0):
            return SOUTH
        else:
            # Not blocked, move to dest!  First move up, then left/right
            if taxi_row == 1:
                return NORTH
            return WEST if taxi_col > dest[1] else EAST

    if taxi_row > 2:
        # If vertically close to destination but blocked by col, move to row 2
        if (taxi_col > 0 and dest[1] == 0) or (taxi_col < 3 and dest[1] == 3):
            return NORTH
        else:
            # Not blocked, move to dest!  First move down, then left/right
            if taxi_row == 3:
                return SOUTH
            return WEST if taxi_col > dest[1] else EAST

    print("No action found!")


def extract_observation(observation):
    destination_id = observation % 4
    destination = {0: RED, 1: GREEN, 2: YELLOW, 3: BLUE}[destination_id]
    passenger_id = (observation // 4) % 5
    passenger = {0: RED, 1: GREEN, 2: YELLOW, 3: BLUE, 4: "taxi"}[passenger_id]
    taxi_col = (observation // 20) % 5
    taxi_row = observation // 100
    return destination, passenger, taxi_row, taxi_col


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


def get_action_frozen_lake_n(observation, info):
    row, col = get_coordinate(observation["position"], observation["board_size"])
    return (row > col) + 1


def get_coordinate(observation, board_size=20):
    return observation // board_size, observation % board_size


def get_action_dollar(observation, info):
    if observation["draw"] < 86:
        return 1
    else:
        return 0


def get_action_number_guessing(observation, info):
    return (observation["low"] + observation["high"]) // 2


def get_interactive_farkle(observation, info: dict) -> dict:
    got_action = False
    while not got_action:
        keep_str = input("Enter keeps, i.e. 1,3,6 will keep 1, 3 and 6: ")
        try:
            keeps = [int(x) for x in keep_str.split(",")]
            keeps = [True if x in keeps else False for x in range(1, 7)]
            got_action = True
        except:  # noqa
            warning(f"Invalid action {keep_str}!")

    got_continue = False
    while not got_continue:
        continue_str = input("Enter continue bool, i.e. stop if 1 and continue with 0")
        try:
            continue_bool = bool(continue_str)
            got_continue = True
        except:  # noqa
            warning(f"Invalid action {continue_str}")

    return {"keep": keeps, "continue": continue_bool}
