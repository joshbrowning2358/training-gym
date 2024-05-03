# 1 -> 1
# 2 -> 1
# 3 -> 2
# 4 -> 3
# 5 -> 5
# 6 -> 8
import sys


def fibonacci_slow(n):
    if n in [1, 2]:
        return 1
    return fibonacci_slow(n - 1) + fibonacci_slow(n - 2)


def fibonacci(n):
    nums = [1, 1]
    while len(nums) < n:
        nums += [nums[-1] + nums[-2]]
    return nums[-1]


sys.set_int_max_str_digits(100000)
fibonacci(100000)
