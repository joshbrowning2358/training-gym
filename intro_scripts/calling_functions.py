import numpy as np


def get_coordinate(position, board_size):
    row = np.floor(position / board_size)
    col = position - board_size * row
    return row, col


# 0  1  2  3  4
# 5  6  7  8  9
# 10 11 12 13 14
# 15 16 17 18 19
# 20 21 22 23 24


get_coordinate(0, 20)


def f(x):
    y = x + 1
    z = 2 * y - 3
    return 4 * z / 7


def split_name(first_last):
    return first_last.split(" ")


first, last = split_name("Breck Woolworth")
first
last

first, middle, last = split_name("Diego Dean Browning")
