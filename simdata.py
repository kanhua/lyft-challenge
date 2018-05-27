import os.path
from glob import glob
import numpy as np
import skimage.io
import skimage.transform
import scipy.misc
import random
import matplotlib.pyplot as plt


def preprocess_images(data_folder, car_hood_mask, image_shape=None, crop_coordiates=None, show_image=False):
    """
    preprocess image and write them to npy files.

    :param show_image:
    :param car_hood_mask:
    :param data_folder:
    :param image_shape: (height, width). None: no resize
    :param crop_coordiates: tuple of (y_min, x_min, y_max, x_max) integer coordinates. None: no cropping
    :return:
    """
    image_paths = glob(os.path.join(data_folder, 'CameraRGB', '*.png'))
    label_paths = glob(os.path.join(data_folder, 'CameraSeg', '*.png'))

    images = []
    gt_images = []

    for i in range(0, len(image_paths), 1):

        image = skimage.io.imread(image_paths[i], format='png')
        gt_image = skimage.io.imread(label_paths[i], format='png')
        # gt_image=gt_image.astype(np.uint8)

        if car_hood_mask is not None:
            gt_image[:,:,0]=gt_image[:,:,0]*car_hood_mask

        if crop_coordiates is not None:
            y_min, x_min, y_max, x_max = crop_coordiates
            image = image[y_min:y_max, x_min:x_max, :]
            gt_image = gt_image[y_min:y_max, x_min:x_max, :]

        if image_shape is not None:
            image = skimage.transform.resize(image, image_shape, preserve_range=True)
            gt_image = skimage.transform.resize(gt_image, image_shape, preserve_range=True)

        image = image.astype(np.uint8)
        gt_image = gt_image.astype(np.uint8)

        assert gt_image.max() > 1

        if i == 0 and show_image:
            plt.imshow(image)
            plt.show()
            plt.figure()
            plt.imshow(gt_image[:,:,0])
            plt.show()

        images.append(image)

        gt_image = gen_one_hot_image(gt_image)

        gt_images.append(gt_image)

    images = np.array(images)
    gt_images = np.array(gt_images)
    np.save("./data/train_data.npy", images)
    np.save("./data/train_label.npy", gt_images)


def gen_one_hot_image(gt_image):
    gt_image = gt_image[:, :, 0]
    gt_road = (gt_image == 7)
    gt_veh = (gt_image == 10)
    gt_bg = np.logical_not(np.logical_or(gt_road, gt_veh))
    # gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
    # gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
    gt_image = np.stack((gt_bg, gt_road, gt_veh), axis=-1)
    return gt_image


class ImageNpy(object):

    def __init__(self, image_file, label_file):
        self.images = np.load(image_file)
        self.labels = np.load(label_file)
        self.data_entry_num = self.images.shape[0]

    def get_batches_fn(self, batch_size):
        data_entries = np.arange(0, self.data_entry_num, 1)
        np.random.shuffle(data_entries)

        for batch_i in range(0, self.data_entry_num, batch_size):
            image_batch = self.images[data_entries[batch_i:batch_i + batch_size], :, :, :]
            label_batch = self.labels[data_entries[batch_i:batch_i + batch_size], :, :, :]

            yield image_batch, label_batch

    def get_bathes_fn_with_crop(self, batch_size, crop_size):
        data_entries = np.arange(0, self.data_entry_num, 1)
        np.random.shuffle(data_entries)
        xw = crop_size[0]  # height
        yw = crop_size[1]  # width
        orig_xw = self.images.shape[1]
        orig_yw = self.images.shape[2]

        start_xw = np.random.randint(0, orig_xw - xw)
        start_yw = np.random.randint(0, orig_yw - yw)

        for batch_i in range(0, self.data_entry_num, batch_size):
            image_batch = self.images[data_entries[batch_i:batch_i + batch_size],
                          start_xw:start_xw + xw, start_yw:start_yw + yw, :]

            label_batch = self.labels[data_entries[batch_i:batch_i + batch_size],
                          start_xw:start_xw + xw, start_yw:start_yw + yw, :]

            yield image_batch, label_batch



if __name__ == "__main__":
    size1 = (448, 448 * 2)
    size2 = None
    car_hood_mask=np.load("hood_mask.npy")
    preprocess_images("./data", car_hood_mask, image_shape=size2, crop_coordiates=(0, 0, 520, 800))

    image_data = ImageNpy("./data/train_data.npy", "./data/train_label.npy")
    for x, y in image_data.get_batches_fn(5):
        #print(x.shape)
        #print(y.shape)
        skimage.io.imsave("testout.png", x[0])
