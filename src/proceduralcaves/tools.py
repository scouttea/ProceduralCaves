import numpy as np


def scale(level, height=2, width=None):
    """
    Scales an array (n x m) to ((height * n) x (width * m))
    if width == None: width will be set equal to height

    >>> import numpy as np
    >>> a = np.array([[2,5],[1,8]])
    >>> a
    array([[2, 5],
           [1, 8]])
    >>> scale(a, height=3, width=2)
    array([[2, 2, 5, 5],
           [2, 2, 5, 5],
           [2, 2, 5, 5],
           [1, 1, 8, 8],
           [1, 1, 8, 8],
           [1, 1, 8, 8]])
    """
    if width is None:
        width = height
    return np.kron(level, np.ones((height, width))).astype(level.dtype)


def border(level, height=2, width=None, copy=False):
    """
    Adds a border to an array (does not modify shape)
    if width == None: width will be set equal to height
    if copy: won't modify the original array

    >>> import numpy as np
    >>> a = np.zeros((6,6))
    >>> a
    array([[0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.]])
    >>> border(a,height=2,width=1)
    array([[1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1.],
           [1., 0., 0., 0., 0., 1.],
           [1., 0., 0., 0., 0., 1.],
           [1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1.]])
    """
    if copy:
        level = level.copy()

    if width is None:
        width = height

    h, w = level.shape

    level[0:height, :] = 1
    level[(h - height):h, :] = 1

    level[:, 0:width] = 1
    level[:, (w - width):w] = 1

    return level
