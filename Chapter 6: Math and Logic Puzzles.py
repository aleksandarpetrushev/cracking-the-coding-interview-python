#6
import math
def cross_off(prime, is_prime):
    i = prime * prime
    while i < len(is_prime):
        is_prime[i] = False
        i += prime
    return is_prime

def next_prime(prime, is_prime):
    next = prime + 1
    while next < len(is_prime) and is_prime[next] == False:
        next += 1
    return next

def sieve(maximum):
    is_prime = [True] * (maximum + 1)

    prime = 2
    while prime <= math.sqrt(maximum):
        is_prime = cross_off(prime, is_prime)
        prime = next_prime(prime, is_prime)

    result = [prime for prime in range(maximum + 1) if is_prime[prime]]
    return result

#6.7 The Apocalypse
import random

def gender_ratio(n_families):
    boys = []
    girls = n_families
    for i in range(n_families):
        current_boys = 0
        while True:
            rand_num = random.random()
            if rand_num < 0.5:
                current_boys += 1
            else:
                break
        boys.append(current_boys)
    boys = sum(boys)
    return boys / (boys + girls)

#6.9 The Egg Drop Problem
def lockers():
    lockers = [False] * 101
    for i in range(1, 101):
        for j in range(i, 101, i):
            lockers[j] = not lockers[j]

    lockers = [val for val in range(1, 101) if lockers[val] == True]
    return lockers
