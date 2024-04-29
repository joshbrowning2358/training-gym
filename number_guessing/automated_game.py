import numpy as np


class GuessingGame:
    def __init__(self, size=100):
        self.answer = np.random.randint(1, size + 1, size=1)[0]
        self.num_guesses = 0
        self.allowed_guesses = np.ceil(np.log2(size)) + 2

    def guess(self, guess: int) -> int:
        if self.num_guesses == self.allowed_guesses:
            raise ValueError("Too many guesses!  You lose!")

        self.num_guesses += 1

        if guess == self.answer:
            print(f"You guessed correctly!  The answer is {guess}!")
            return 0
        elif guess > self.answer:
            print(f"You guessed {guess} but that's too high!")
            return 1
        else:
            print(f"You guessed {guess} but that's too low!")
            return -1
