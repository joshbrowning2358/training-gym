import numpy as np


def run_game():
    answer = np.random.randint(1, 101, size=1)[0]
    done = False
    while not done:
        guess = int(input("Enter your guess: "))
        if guess == answer:
            print("You guessed correctly!")
            done = True
        elif guess > answer:
            print("Your guess is too high!")
        else:
            print("Your guess is too low!")


if __name__ == "__main__":
    run_game()
