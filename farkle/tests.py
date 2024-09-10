import numpy as np

from farkle import Farkle


def test_scoring():
    assert Farkle.score(np.array([3] * 6)) == (3000, True)
    assert Farkle.score(np.array([5] * 6)) == (3000, True)

    assert Farkle.score(np.array([2] * 5)) == (2000, True)
    assert Farkle.score(np.array([6] * 5)) == (2000, True)

    assert Farkle.score(np.array([1] * 4)) == (1000, True)

    assert Farkle.score(np.array([1, 2, 3, 4, 5, 6])) == (1500, True)
    assert Farkle.score(np.array([3, 5, 2, 1, 4, 6])) == (1500, True)
    assert Farkle.score(np.array([1, 1, 4, 4, 3, 3])) == (1500, True)
    assert Farkle.score(np.array([3, 4, 4, 3, 4, 3])) == (2500, True)

    assert Farkle.score(np.array([6, 1, 6, 6, 5])) == (750, True)
    assert Farkle.score(np.array([5, 1, 5, 5])) == (600, True)

    assert Farkle.score(np.array([4])) == (0, False)
    assert Farkle.score(np.array([1, 1, 1, 3])) == (300, False)
    assert Farkle.score(np.array([4, 2, 4, 4])) == (400, False)

    assert Farkle.score(np.array([])) == (0, False)
