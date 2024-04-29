from number_guessing.automated_game import GuessingGame

game = GuessingGame(size=10)
game.guess(9)
game.guess(8)
game.guess(7)
game.guess(6)
game.guess(5)
game.guess(4)


game = GuessingGame(size=10)
for i in range(1, 10):
    game.guess(i)
