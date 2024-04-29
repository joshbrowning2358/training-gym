# Problem: find the first N primes, so maybe first 1000 primes.
import numpy as np


def find_first_primes(n):
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for i in primes:
            if np.floor(candidate / i) == candidate / i:
                is_prime = False
                break
            if i > np.sqrt(candidate):
                break
        if is_prime:
            primes = primes + [candidate]
        candidate += 1

    return primes


find_first_primes(10000)
