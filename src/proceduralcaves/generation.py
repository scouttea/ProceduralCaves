import numpy as np
from numba import njit
from scipy.signal import convolve2d


# Jitting might not be usefull
@njit()
def random_noise(height, width, fill=0.5, seed=125):
    """
    Returns a random boolean array of size (height x width)
    where $fill$ of cells are True / 1

    >>> random_noise(5, 8, seed=125).astype("int")
    array([[0, 1, 0, 1, 1, 0, 0, 1],
           [0, 1, 1, 1, 0, 1, 0, 0],
           [0, 0, 1, 1, 0, 0, 0, 1],
           [1, 1, 0, 1, 1, 1, 0, 1],
           [1, 0, 1, 1, 0, 0, 0, 1]])
    """
    np.random.seed(seed)
    level = np.random.rand(height, width) < fill
    return level


# Can't jit without numba compatible Convolve2d function
def cellular_automata(level, lower=3, regen=4, area=1, iteration=8):
    """
    Applies a simple cellular_automata on the level
    This is a way of generating smooth looking caves from noise

    Each iteration apply the following rule
        if (i,j) is active and so are more than $lower$ of its neighbours
            then keep (i,j) active
            else make (i,j) inactive
        if (i,j) is inactive but more than $regen$ of its neighbours are active
            then make (i,j) active
            else keep (i,j) inactive
    Active means level[i,j] == 1, and neighbours are all cells within distance
    $area$ (horizontal, vertical and diagonal, a square area around)

    Depending on the area varying amounts of iterations will be needed
    However usually not a lot will be needed
     """
    kernel = np.ones((1, area*2+1))

    # calculate neighbours using kernel convolution
    # 1 ... 1 ... 1
    # 1 ... 0 ... 1
    # 1 ... 1 ... 1
    # Which can be calculated using a kernel of all ones and
    # subtracting the original level from it
    # 1 ... 1 ... 1     1 ... 1 ... 1     0 ... 0 ... 0
    # 1 ... 0 ... 1  =  1 ... 1 ... 1  -  0 ... 1 ... 0
    # 1 ... 1 ... 1     1 ... 1 ... 1     0 ... 0 ... 0

    for _ in range(iteration):
        # all ones kernel is seperatable
        convolve = convolve2d(level, kernel, fillvalue=1, mode="same")
        convolve = convolve2d(convolve, kernel.T, fillvalue=1, mode="same")

        nbors = convolve - level

        # apply rules
        level = level & (nbors > lower) | (level == 0) & (nbors > regen)

    return level
