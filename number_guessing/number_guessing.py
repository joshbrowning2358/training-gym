import numpy as np


def run_game():
    print("Welcome to the number guessing game!")
    answer = np.random.randint(1, 101, size=1)[0]
    guess = int(input("Enter your guess: "))

    if guess < answer:
        print("your guess was to small, guess again LOSER")
    if guess > answer:
        print("your guess was too high guess again you dumb weirdo")
    if guess == answer:
        print("you got lucky, in your face breck and mike and josh oh and diego i hope you get better, NOT")


if __name__ == "__main__":
    run_game()
