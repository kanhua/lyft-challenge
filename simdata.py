import os.path
from glob import glob
import numpy as np
import skimage.io
import skimage.transform
import scipy.misc
import random
import matplotlib.pyplot as plt
import warnings

_CAR_INDEX = 2

UPPER_CUT = 170
BOTTOM_CUT=520

CLASS_WEIGHT = np.array([[2.047828, 3.52548263, 10.25242643]])


def preprocess_images(data_folder, car_hood_mask, image_shape=None, crop_coordiates=None, show_image=False,
                      car_pixel_threshold=0, image_file="./data/train_data.npy", label_file="./data/train_label.npy"):
    """
    preprocess image and write them to npy files.

    :param label_file:
    :param image_file:
    :param car_pixel_threshold:
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

    data_entries=1000

    shuffuled_index=np.random.permutation(len(image_paths))


    for i in shuffuled_index:

        image = skimage.io.imread(image_paths[i], format='png')
        gt_image = skimage.io.imread(label_paths[i], format='png')
        # gt_image=gt_image.astype(np.uint8)

        if car_hood_mask is not None:
            gt_image[:, :, 0] = gt_image[:, :, 0] * car_hood_mask

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
            plt.imshow(gt_image[:, :, 0])
            plt.show()

        car_label = 10
        car_pixels = np.sum(gt_image[:, :, 0] == car_label)

        if car_pixels >= car_pixel_threshold:

            gt_image = gen_one_hot_image(gt_image)

            images.append(image)

            gt_images.append(gt_image)
            if show_image and np.random.rand() < 0.1:
                skimage.io.imsave("./figures/image_{}.png".format(i), image)

        #if len(images)>=data_entries:
        #    break

    images = np.array(images)
    gt_images = np.array(gt_images)
    np.save(image_file, images)
    np.save(label_file, gt_images)


def gen_one_hot_image(gt_image):
    gt_image = gt_image[:, :, 0]
    gt_road = np.logical_or((gt_image == 7), (gt_image == 6))
    gt_veh = (gt_image == 10)
    gt_bg = np.logical_not(np.logical_or(gt_road, gt_veh))
    # gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
    # gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
    gt_image = np.stack((gt_bg, gt_road, gt_veh), axis=-1)
    return gt_image


def shift_hue(rgb_image, mask, supress_warning=True):
    with warnings.catch_warnings():
        if supress_warning:
            warnings.simplefilter("ignore")
        hsv_image = skimage.color.rgb2hsv(rgb_image)

    shifted_h = np.fmod(hsv_image[:, :, 0] + np.random.rand(), 1)

    hsv_image[:, :, 0] += (shifted_h * mask)

    return skimage.color.hsv2rgb(hsv_image)


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

    def filter_low_vehicle(self):
        data_entries = np.arange(0, self.data_entry_num, 1)
        np.random.shuffle(data_entries)

        train_data_entries = []

        threshold = 5000
        select_prob = 0.3
        image_count = 0
        idx = 0
        while image_count < self.labels.shape[0]:
            idx = idx % len(data_entries)
            car_count = np.sum(self.labels[data_entries[idx], :, :, _CAR_INDEX])
            if car_count < threshold:
                if np.random.rand() < select_prob:
                    train_data_entries.append(data_entries[idx])
                    image_count += 1
            else:
                train_data_entries.append(data_entries[idx])
                image_count += 1
            idx += 1
        return np.array(train_data_entries)

    def get_bathes_fn_with_crop(self, batch_size, crop_size, shift_hue_prob=1, shuffle=True, filter=False):
        data_entries = np.arange(0, self.data_entry_num, 1)
        if shuffle:
            np.random.shuffle(data_entries)

        if filter:
            data_entries = self.filter_low_vehicle()

        orig_xw = self.images.shape[1]
        orig_yw = self.images.shape[2]

        if crop_size is not None:
            xw = crop_size[0]  # height
            yw = crop_size[1]  # width
            start_xw = np.random.randint(0, orig_xw - xw)
            start_yw = np.random.randint(0, orig_yw - yw)
        else:
            xw = orig_xw
            yw = orig_yw
            start_xw = 0
            start_yw = 0

        for batch_i in range(0, self.data_entry_num, batch_size):
            image_batch = self.images[data_entries[batch_i:batch_i + batch_size],
                          start_xw:start_xw + xw, start_yw:start_yw + yw, :]

            label_batch = self.labels[data_entries[batch_i:batch_i + batch_size],
                          start_xw:start_xw + xw, start_yw:start_yw + yw, :]

            if np.random.rand() < shift_hue_prob:
                print("shift hue...")
                for image_index in range(image_batch.shape[0]):
                    conv_image = shift_hue(image_batch[image_index], label_batch[image_index, :, :, _CAR_INDEX])
                    image_batch[image_index] = (conv_image * 256).astype(np.uint8)

            yield image_batch, label_batch

def gen_validation_data():

    val_data_path="/Users/kanhua/Downloads/data/Valid"
    car_hood_mask = np.load("hood_mask.npy")
    preprocess_images(val_data_path, car_hood_mask, image_shape=None, crop_coordiates=(UPPER_CUT, 0, BOTTOM_CUT, 800),
                      image_file="./data/val_data.npy",label_file="./data/val_label.npy")


def gen_general_train_data():

    val_data_path="/Users/kanhua/Downloads/data/Train"
    car_hood_mask = np.load("hood_mask.npy")
    preprocess_images(val_data_path, car_hood_mask, image_shape=None, crop_coordiates=(UPPER_CUT, 0, BOTTOM_CUT, 800),
                      image_file="./data/train_data_2.npy",label_file="./data/train_label_2.npy")


def gen_train_data():

    size2 = None
    car_hood_mask = np.load("hood_mask.npy")
    preprocess_images("./data", car_hood_mask, image_shape=size2, crop_coordiates=(UPPER_CUT, 0, BOTTOM_CUT, 800),
                      image_file="./data/train_data_baseline.npy",label_file="./data/train_label_baseline.npy")


if __name__ == "__main__":

    #gen_validation_data()

    #gen_general_train_data()

    gen_train_data()


