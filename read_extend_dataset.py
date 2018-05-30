
from simdata import preprocess_images
import numpy as np

size1 = (448, 448 * 2)
size2 = None
car_hood_mask = np.load("hood_mask.npy")

data_folder="/Users/kanhua/Downloads/data/Train/"
preprocess_images(data_folder, car_hood_mask, image_shape=size2, crop_coordiates=(170, 0, 520, 800),
                  car_pixel_threshold=10000,show_image=True)