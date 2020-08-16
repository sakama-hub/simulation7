# -*- coding utf-8 -*-
"""add the nonliner phase noises."""


def add_noise(X, lw, fre):
    """
    Fuction to add noise to symbols.

    input
    X:sequences
    lw:linewidth of Tx or Lo
    fre:sampling frequency
    -----------------------------
    output
    Y:symbol added noise
    """
    import numpy as np
    import math
    import cmath

    num_sy = len(X)
    dis = 2 * cmath.pi * lw / fre
    noi = math.sqrt(dis) * np.random.normal(loc=0, scale=1, size=num_sy)
    # pn = np.cumsum(noi)
    Y = [complex(X[i, 0], X[i, 1])*cmath.exp(complex(0, math.pi*noi[i])) for i in range(num_sy)]
    Y = np.array([[Y_.real, Y_.imag]for Y_ in Y])
    return Y
