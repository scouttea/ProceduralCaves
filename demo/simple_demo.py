from proceduralcaves.generation import random_noise, cellular_automata

from PIL import Image
import numpy as np

# Parameters
seed = 1337
lower = 3
regen = 4
area = 1
iteration = 8

# Generation
level = random_noise(height=128, width=128, seed=seed)
level = cellular_automata(level,
                          lower=lower,
                          regen=regen,
                          area=area,
                          iteration=iteration)

# Saving data
# invert such that 1=black
img = Image.fromarray(np.uint8((1 - level) * 255), 'L')
img.save('simple_demo_result.png')
