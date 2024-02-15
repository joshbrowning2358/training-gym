import numpy as np


def run_game():
    print("Welcome to the number guessing game!")
    answer = np.random.randint(1, 101, size=1)[0]
    guess = int(input("Enter your guess: "))

    if guess == answer:
        print("You win!")
    if guess < answer:
        print("guess lower")
    if guess > answer:
        print("guess higher")


if __name__ == "__main__":
    run_game()
