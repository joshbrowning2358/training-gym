import numpy as np


def run_game():
    print("Welcome to the number guessing game!")
    answer = np.random.randint(1, 101, size=1)[0]
    guess = input("Enter your guess: ")
    print("You guessed", guess, "but the answer is", answer)


if __name__ == "__main__":
    run_game()
