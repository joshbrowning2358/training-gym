import numpy as np
from gymnasium import Env
from gymnasium.spaces import Dict, Discrete, Tuple

# How many points do you need to win Farkle?
WINNING_THRESHOLD = 10000


class Farkle(Env):
    def __init__(self):
        super(Farkle, self).__init__()

        space = {
            "keep": Tuple((Discrete(2), Discrete(2), Discrete(2), Discrete(2), Discrete(2), Discrete(2))),
            "stop": Discrete(2),
        }
        self.action_space = Dict(space)

        # TODO: Add observation for other player's score
        self.observation_space = Dict(
            {
                "current_round_score": Discrete(10000),
                "current_score": Discrete(10000),
                "current_keep": Tuple((Discrete(2), Discrete(2), Discrete(2), Discrete(2), Discrete(2), Discrete(2))),
                "dice": Tuple((Discrete(6), Discrete(6), Discrete(6), Discrete(6), Discrete(6), Discrete(6))),
            }
        )

        self.current_score = 0
        self.current_round_score = 0
        self.current_keep = (0, 0, 0, 0, 0, 0)
        self.dice = np.random.randint(1, 7, 6)

    def reset(self) -> tuple[dict, dict]:
        self.current_score = 0
        self.current_round_score = 0
        self.current_keep = (0, 0, 0, 0, 0, 0)
        self.dice = np.random.randint(1, 7, 6)
        return {"current_score": 0, "current_round_score": 0, "current_keep": (0, 0, 0, 0, 0, 0), "dice": self.dice}, {}

    def render(self, mode="human"):
        print(f"Your current score is {self.current_score}")

    def step(self, action: dict) -> tuple[dict, int | float, bool, bool, dict]:
        """
        Returns:
            Tuple of observation, reward, terminated, truncated, info
        """
        if action["stop"] == 0:
            return self._move_keep_rolling(action)
        else:
            return self._move_stop_rolling()

    def _move_keep_rolling(self, action: dict) -> tuple[dict, int | float, bool, bool, dict]:
        """
        Continuing to roll and keeping dice.  Must check that it's a valid set of dice to keep,
        otherwise we need to return a reward of 0 and end the turn.

        Args:
            action: Dict with key "keep" specifying which dice should be kept.

        Returns:
            Tuple of observation, reward, terminated, truncated, info
        """
        # Continuing to roll, must assert that we kept dice worth something
        kept_filter = [new == 1 and old == 0 for new, old in zip(action["keep"], self.current_keep)]
        dice_to_score = np.array([die for die, keep in zip(self.dice, kept_filter) if keep])
        score, used_all = self.score(dice_to_score)
        if (score == 0) or (not used_all):
            # Invalid move, tried to
            #   a) keep playing without dice worth anything
            #   b) didn't use all the dice they tried to keep
            reward = 0
            return self.end_turn(), reward, False, False, {}
        else:
            return self._valid_move_and_continuing(score, action)

    def _move_stop_rolling(self) -> tuple[dict, int | float, bool, bool, dict]:
        """
        Ending turn and keeping current score.

        Returns:
            Tuple of observation, reward, terminated, truncated, info
        """
        new_roll_filter = [keep == 0 for keep in self.current_keep]
        dice_to_score = np.array([die for die, keep in zip(self.dice, new_roll_filter) if keep])
        score, _ = self.score(dice_to_score)
        self.current_round_score += score
        if score == 0:
            reward = 0
            return self.end_turn(), reward, False, False, {}
        else:
            self.current_score += self.current_round_score
            if self.current_score > WINNING_THRESHOLD:
                reward = 100
                return self.end_turn(), reward, True, False, {}

            # Scale the reward down a bit.  We want to reward the algo for scoring points, but
            # not too much.  Really reward it once we win the game.
            reward = self.current_round_score / 1000
            self.current_round_score = 0
            self.current_keep = (0, 0, 0, 0, 0, 0)
            return self.end_turn(), reward, False, False, {}

    def _valid_move_and_continuing(self, score: float, action: dict) -> tuple[dict, int | float, bool, bool, dict]:
        """
        Valid move, adding score to current score and rolling again.  Part of the move_keep_rolling
        path.

        Args:
            score: The computed score of the kept dice.
            action: Dict with key "keep" specifying which dice should be kept.

        Returns:
            Tuple of observation, reward, terminated, truncated, info
        """
        self.current_round_score += score
        self.current_keep = [old or new for old, new in zip(self.current_keep, action["keep"])]
        keep_arr = np.array(self.current_keep)
        if all(keep_arr):
            # All dice have been kept, need to roll again
            self.current_keep = (0, 0, 0, 0, 0, 0)
            self.dice = np.random.randint(1, 7, 6)
        else:
            # Reroll only non-kept dice
            self.dice[keep_arr == 0] = np.random.randint(1, 7, sum(1 - keep_arr))
        return (
            {
                "current_score": self.current_score,
                "current_round_score": score,
                "current_keep": self.current_keep,
                "dice": self.dice,
            },
            score,
            False,
            False,
            {},
        )

    def end_turn(self) -> dict:
        """
        Updates the current state assuming a new roll and returns a state dictionary representing that.
        """
        self.dice = np.random.randint(1, 7, 6)
        self.current_round_score = 0
        self.current_keep = (0, 0, 0, 0, 0, 0)
        return {
            "current_score": self.current_score,
            "current_round_score": 0,
            "current_keep": (0, 0, 0, 0, 0, 0),
            "dice": self.dice,
        }

    @classmethod
    def score(cls, dice: np.ndarray) -> tuple[int, bool]:
        """
        Args:
            dice: A numpy array of dice rolls.  Can be fewer than 6 as subsequent rounds need to also be scored.

        Returns:
            Tuple containing:
                The score of the dice
                A boolean indicating if all the dice were used in scoring
        """
        score, n_remaining = cls.score2(dice)
        return score, n_remaining > 0

    @staticmethod
    def score2(dice: np.ndarray) -> tuple[int, int]:
        """
        Args:
            dice: A numpy array of dice rolls.  Can be fewer than 6 as subsequent rounds need to also be scored.

        Returns:
            Tuple containing:
                The score of the dice
                The number of remaining dice
        """
        if len(dice) == 0:
            return 0, 0

        uniqs, cnts = np.unique(dice, return_counts=True)
        if len(uniqs) == 6:  # Straight
            return 1500, 0
        if len(uniqs) == 3 and all(cnts == 2):  # 3 pair
            return 1500, 0
        if len(uniqs) == 2 and set(cnts) == {2, 4}:  # 4 of a kind and a pair
            return 1500, 0
        if len(uniqs) == 2 and all(cnts == 3):  # 2 3 of a kind
            return 2500, 0
        if max(cnts) == 6:  # 6 of a kind
            return 3000, 0

        # Any further scoring may be partial, so we'll need to remove 5/4/3 of a kind and score remaining
        score = 0
        if cnts.max() == 5:  # 5 of a kind
            score += 2000
            uniqs = uniqs[cnts != 5]
            cnts = cnts[cnts != 5]
            if len(cnts) == 0:
                return score, 0
        if cnts.max() == 4:  # 4 of a kind
            score += 1000
            uniqs = uniqs[cnts != 4]
            cnts = cnts[cnts != 4]
            if len(cnts) == 0:
                return score, 0
        if cnts.max() == 3:  # 3 of a kind
            dice_val = uniqs[cnts == 3][0]
            score += dice_val * 100 + (dice_val == 1) * 200  # 1's -> 300, all else is 100 * dice value
            uniqs = uniqs[cnts != 3]
            cnts = cnts[cnts != 3]
            if len(cnts) == 0:
                return score, 0

        # Remaining scoring dice are 1's and 5's
        score += 100 * cnts[uniqs == 1].sum() + 50 * cnts[uniqs == 5].sum()
        uniqs = uniqs[(uniqs != 1) & (uniqs != 5)]

        return score, len(uniqs)
