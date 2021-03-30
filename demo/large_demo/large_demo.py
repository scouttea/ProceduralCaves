from proceduralcaves.generation import random_noise, cellular_automata
from proceduralcaves.tools import scale, border

from PIL import Image
import numpy as np


def save_image(level, name):
    img = Image.fromarray(np.uint8((1 - level) * 255), 'L')
    img.save(name + '.png')


size = np.array((64, 64))
seed = 415

level = random_noise(*size, seed=seed)
save_image(level, "1_starting_noise")

level = cellular_automata(level, 3, 4, 1, iteration=4)
save_image(level, "2_large_scale_profile")

# mask = random_noise(seed, *size)
# level = mask & level

level = border(level, height=2)
save_image(level, "3_border")

level = scale(level, height=2, width=4)
mask = random_noise(*level.shape, seed=seed)
level = level | mask
save_image(level, "4_upscale_noise")

level = cellular_automata(level, 13, 20, 2, iteration=1)
save_image(level, "5_medium_scale_profile")

level = scale(level, 2)
mask = random_noise(*level.shape, seed=seed)
level = level | mask
save_image(level, "6_upscale_noise")

level = cellular_automata(level, 15, 20, 2, iteration=1)
level = cellular_automata(level, 3, 5, 1, iteration=2)
save_image(level, "7_small_scale_profile")
np.save("small_detail_level_large_demo", level)  # save result for other demos


# Detail steps should be skipped when adding details some other way
level = scale(level, 2)
mask = random_noise(*level.shape, seed=seed)
level = level | mask
save_image(level, "8_upscale_noise")

level = cellular_automata(level, 12, 16, 2, iteration=1)
level = cellular_automata(level, 3, 5, 1, iteration=8)
save_image(level, "9_detail_scale_profile")
