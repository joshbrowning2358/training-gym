import numpy as np


def build_square(n: int, validate: bool = True):
    square = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            square[i][j] = (i + j) % n + 1
    square = square[np.random.permutation(n)]
    square = square[:, np.random.permutation(n)]

    if validate:
        for i in range(n):
            assert len(np.unique(square[i])) == n
        for j in range(n):
            assert len(np.unique(square[:, j])) == n
    return square.astype(str)


def solve(square: np.ndarray, max_iterations: int = 100):
    n = square.shape[0]
    n_iterations = 0
    while np.any(square == "."):
        for i in range(n):
            for j in range(n):
                if square[i, j] == ".":
                    options = {str(x) for x in range(1, n + 1)}
                    for x in set(square[i]):
                        options.discard(x)
                    for x in set(square[:, j]):
                        options.discard(x)
                    if len(options) == 1:
                        square[i, j] = options.pop()
        n_iterations += 1
        if n_iterations > max_iterations:
            raise ValueError(f"Could not solve the square in {max_iterations} iterations.")
    return square


latin_square = [
    ["2", ".", "1", "4"],
    ["4", ".", "2", "3"],
    ["1", "4", ".", "."],
    ["3", ".", ".", "1"]
]
solve(np.array(latin_square))


def generate_puzzle(n: int, to_delete: int = 3):
    square = build_square(n)
    best = square
    while True:
        rows = np.random.choice(n, size=to_delete)
        cols = np.random.choice(n, size=to_delete)
        square[rows, cols] = "."
        try:
            solve(square.copy())
            best = square.copy()
            print(best)
        except ValueError:
            break

    return best


latin_square = generate_puzzle(100, to_delete=10)
solve(latin_square.copy())
