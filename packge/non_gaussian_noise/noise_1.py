# -*- coding utf-8 -*-
"""function to add noise to symbols."""
from random import gauss  # generate discrete value according to Gauss distribution
import math  # function module to deal with real
import cmath  # function module to deal with complex
import numpy as np
import random

def add_noise(symbols, CNR, num=16):
    """INPUT
       Symbols : array of array that includes I valuable and Q valuable,
       CNR : carrier noise ratio,
       num : the number of symbols that  modulater can express
       ----------------------------------------------------------
       OUTPUT
       symbol_with_noise : datatype is array of array. The list includes I valuable and Q valuable added noise
       ----------------------------------------------------------
       Note
       noise = noise=exp(1i*pi/4)*(randn(10000,1)+0.7*1i*randn(10000,1)
        """

    #  difine the power of carrer according to input
    if num == 4:
        pow_av = 2
    elif num == 8:
        pow_av = 3
    else:
        pow_av = 10
    noises = np.array([cmath.exp(1j * math.pi / 4) * (gauss(0.0, 1.0)) + 0.7
                       * 1j * (gauss(0.0, 1.0)) for _ in range(len(symbols))])
    noise_pow_av = 1.49
    noise_pow_control = pow(10, (10 * math.log10(pow_av / noise_pow_av) - CNR) / 20)
    symbol_with_noise = symbols + np.array([[noise.real, noise.imag] for noise in noises * noise_pow_control])
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
