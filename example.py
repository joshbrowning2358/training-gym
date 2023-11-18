ACTIONS = {
    0: "Do Nothing",
    1: "Fire",
    2: "Move Right",
    3: "Move Left",
    4: "Move Right + Fire",
    5: "Move Left + Fire",
}


def get_action_zero(observation, lives, frame_number):
    return 0


def get_action_shoot(observation, lives, frame_number):
    return 1
