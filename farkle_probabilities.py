import numpy as np

n_sims = 1000000

rolls = np.random.randint(1, 6 + 1, size=(n_sims, 6))
# Remove rolls with 1's or 6's (we'll assume 6's are worth value instead of 5's without lack of generality)
rolls = rolls[rolls.min(axis=1) > 1]
rolls = rolls[rolls.max(axis=1) < 6]
print(f"Probability of scoring no points from 1's and 5's is estimated as {round(rolls.shape[0] / n_sims * 100, 3)}%")


def has_no_points(x: np.ndarray) -> bool:
    uniqs, cnts = np.unique(x, return_counts=True)
    if uniqs.shape[0] <= 3:
        # 3 unique vals => 3 pairs or at least one 3 of a kind
        return False
    elif cnts.max() >= 3:
        return False
    return True


has_no_points(np.array([2, 2, 3, 4, 5, 5]))
has_no_points(np.array([2, 3, 5, 5, 4, 5]))

rolls = rolls[np.apply_along_axis(has_no_points, 1, rolls)]
print(f"Probability of scoring no points is estimated as {round(rolls.shape[0] / n_sims * 100, 3)}%")


# Theoretical Approach:
# 6^6 possiible rolls
# A losing roll cannot contain a 1, 5, 3 pair, three of a kind, straight, four of a kind, five of a kind, six of a kind.
# We only care about 1, 5, 3 pair, three of a kind.  All others would also be detected
# First, a losing roll must contain only 2, 3, 4, 6.
# By way of contradiction, suppose it contains only 3 unique dice.  Then, there must be 2 of each dice (a 3-pair) or one
# die must have 3 (a 3-of-a-kind) and so there are points.  This contradicts, so there must be all 4 dice.
# So, we must have all 4 dice.  Then, we have 2 remaining dice to allocate as we please: 2/3, 2/4, 2/6, etc.  There are
# 6 ways to do this (as we cannot do 2/2).
# However, any of these 6 dice rolls (say 2, 2, 3, 4, 4, 6) is not a single "roll" as we must consider possible
# permutations.  Of these, there are 6 choose 2 choose 2 choose 1 choose 1 = 6! / 2! / 2! = 180.
# Thus, there are 6 * 180 bad rolls and 6^6 total rolls => 6 * 180 / 6^6 = 2.315%.