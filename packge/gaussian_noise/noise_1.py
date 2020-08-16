# -*- coding utf-8 -*-
"""function to add noise to symbols."""

import random
from random import gauss
import math
import numpy as np


def add_noise(symbols, CNR, num=16):
    """INPUT.
       Symbols : array of array that includes I valuable and Q valuable,
       CNR : carrier noise ratio,
       num : the number of symbols that  modulater can express
       ----------------------------------------------------------
       OUTPUT
       symbol_with_noise : datatype is array of array. The list includes I valuable and Q valuable added noise
    """

    #  difine the power of carrer according to input
    if num == 4:
        pow_av = 2
    elif num == 8:
        pow_av = 3
    else:
        pow_av = 10

    noise_pow_av = 2
    noise_pow_control = pow(10, (10 * math.log10(pow_av / noise_pow_av) - CNR) / 20)

    symbol_with_noise = symbols + noise_pow_control * np.array([[gauss(0, 1), gauss(0, 1)] for _ in range(len(symbols))])
    return symbol_with_noise


if __name__ == "__main__":
    CNR = 10
    num_ld = 10000
    a = [1, -1, 3, -3]
    n = 1
    shaping_points = {}
    for i in a:
        for t in a:
            shaping_points["s"+str(n)] = np.array([i, t])
            n += 1

    symbol_labels = random.choices(list(shaping_points.keys()), k=num_ld)
    symbol_points = np.array([shaping_points[key] for key in symbol_labels])
    symbol_points = add_noise(symbol_points, CNR, num=16)

    draw_graph("tutorial.html", shaping_points.values(), symbol_points)
