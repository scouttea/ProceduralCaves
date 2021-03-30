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


@njit
def erode(level, iteration=20000, d=40, aspect=4, momentum=0.6, seed=125):
    """
    Erodes terrain using random walks
        iterations ~ amount of erosion starting locations
        d          ~ how big the each erosion will get
        aspect     ~ ratio between width / height of erosion
        momentum   ~ how likely the erosion will move in 1 direction
    """

    np.random.seed(seed)

    h, w = level.shape

    for _ in range(iteration):
        # generate random starting location
        x = np.random.randint(w)
        y = np.random.randint(h)

        mx, my = 0, 0

        # if starting location isn't empty abort
        # this is to avoid generating seperated pockets
        if level[y, x]:
            continue

        for _ in range(d):
            p = np.random.rand()
            if np.random.rand() > 1/(1 + aspect):
                dir = p < (0.5 + (mx)*momentum)
                x += (dir*2 - 1)  # [0 - 1] -> [-1 1]
                mx = (dir*2 - 1) + mx/2
            else:
                dir = p < (0.5 + (my)*momentum)
                y += (dir*2 - 1)
                my = (dir*2 - 1) + my/2

            # only erode within bounds
            if (y % h == y and x % w == x):
                level[y, x] = 0

    return level
