from functools import reduce

import pandas as pd
import numpy as np
from tqdm import tqdm

from farkle.farkle import Farkle

individual_rolls = [pd.DataFrame({f"roll_{i}": [1, 2, 3, 4, 5, 6]}) for i in range(1, 7)]
rolls = reduce(lambda x, y: pd.merge(x, y, how="cross"), individual_rolls)

total_score = 0
zero_cnt = 0
for _, roll in rolls.iterrows():
    score = Farkle.score(roll.array)[0]
    total_score += score
    zero_cnt += score == 0

avg_score = total_score / rolls.shape[0]
lose_prob = zero_cnt / rolls.shape[0]
print("Average score of non-losing rolls is ", round(avg_score, 2))
print(f"Percent of time you lose when rolling 6 dice: {round(lose_prob * 100, 4)}%")
# E(rolling 6 dice) = P(roll points) * AvgScore + P(roll 0 points) * (- CurrScore)
# CurrScore * P(roll 0 points) = P(roll points) * AvgScore
cutoff_6 = np.ceil((1 - lose_prob) * avg_score / lose_prob / 50) * 50
print(f"You should not roll 6 dice if you have {cutoff_6} pts")

value_func = {
    1: {int(cutoff_6): int(cutoff_6)},
    2: {int(cutoff_6): int(cutoff_6)},
    3: {int(cutoff_6): int(cutoff_6)},
    4: {int(cutoff_6): int(cutoff_6)},
    5: {int(cutoff_6): int(cutoff_6)},
    6: {int(cutoff_6): int(cutoff_6)},
}

def get_rolls(n_dice: int) -> pd.DataFrame:
    return_df = rolls.copy()
    for missing_dice in range(n_dice + 1, 6 + 1):
        return_df = return_df[return_df[f"roll_{missing_dice}"] == 6]
    return return_df[[f"roll_{i}" for i in range(1, n_dice + 1)]]


to_explore = range(int(cutoff_6), 0, -50)
to_explore = [16350, 16300]
for curr_value in tqdm(to_explore):
    for n_dice in [6, 5, 4, 3, 2, 1]:
        dice_rolls = get_rolls(n_dice)
        total = 0
        for _, dice_roll in dice_rolls.iterrows():
            score, n_remaining = Farkle.score2(dice_roll)
            if score > 0:
                if curr_value + score > cutoff_6:
                    total += curr_value + score
                else:
                    n_remaining = n_remaining if n_remaining > 0 else 6
                    total += value_func[n_remaining][int(curr_value + score)]
        value_func[n_dice][curr_value] = int(total / dice_rolls.shape[0])
