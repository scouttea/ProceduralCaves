from proceduralcaves.generation import erode, cellular_automata
from proceduralcaves.tools import scale

import numpy as np
from PIL import Image


seed = 125

level = np.load("small_scale_large_demo.npy")
level = scale(level, 2)

level = erode(level, seed=seed, iteration=10000, d=40)
level = cellular_automata(level, 3, 5, 1, iteration=8)


img = Image.fromarray(np.uint8((1 - level) * 255), 'L')
img.save('erosion_demo_result.png')
