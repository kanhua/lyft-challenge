import os.path
from glob import glob
import numpy as np
import skimage.io
import scipy.misc
import random


def preprocess_images(data_folder, image_shape):
    image_paths = glob(os.path.join(data_folder, 'CameraRGB', '*.png'))
    label_paths = glob(os.path.join(data_folder, 'CameraSeg', '*.png'))

    images = []
    gt_images = []

    for i in range(0, len(image_paths), 1):
        image = scipy.misc.imresize(skimage.io.imread(image_paths[i]), image_shape)

        images.append(image.astype(dtype=np.uint8))

        gt_image = scipy.misc.imresize(skimage.io.imread(label_paths[i]), image_shape)

        gt_image = gt_image[:, :, 0]

        gt_road = (gt_image == 7)
        gt_veh = (gt_image == 10)
        gt_bg = np.logical_not(np.logical_or(gt_road, gt_veh))
        # gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
        # gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
        gt_image = np.stack((gt_bg, gt_road, gt_veh), axis=-1)

        gt_images.append(gt_image)

    images = np.array(images)
    gt_images = np.array(gt_images)
    np.save("./data/train_data.npy", images)
    np.save("./data/train_label.npy", gt_images)


class ImageNpy(object):

    def __init__(self, image_file, label_file):
        self.images = np.load(image_file)
        self.labels = np.load(label_file)
        self.data_entry_num = self.images.shape[0]

    def get_batches_fn(self, batch_size):
        data_entries=np.arange(0, self.data_entry_num, 1)
        np.random.shuffle(data_entries)

        for batch_i in range(0, self.data_entry_num, batch_size):
            image_batch = self.images[data_entries[batch_i:batch_i + batch_size], :, :, :]
            label_batch = self.labels[data_entries[batch_i:batch_i + batch_size], :, :, :]

            yield image_batch, label_batch


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """

    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'CameraRGB', '*.png'))
        label_paths = {
            os.path.basename(path): path
            for path in glob(os.path.join(data_folder, 'CameraSeg', '*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i + batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(skimage.io.imread(gt_image_file), image_shape)

                gt_image = gt_image[:, :, 0]

                gt_road = (gt_image == 7)
                gt_veh = (gt_image == 10)
                gt_bg = np.logical_not(np.logical_or(gt_road, gt_veh))
                # gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                # gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
                gt_image = np.stack((gt_bg, gt_road, gt_veh), axis=-1)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)

    return get_batches_fn


if __name__ == "__main__":
    preprocess_images("./data", (224, 224))

    image_data = ImageNpy("./data/train_data.npy", "./data/train_label.npy")
    for x, y in image_data.get_batches_fn(5):
        print(x.shape)
        print(y.shape)
