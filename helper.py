import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
from skimage import transform as sktf
import skimage.io
import skimage.transform
from PIL import Image
from io import BytesIO, StringIO
import sys, base64




class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


maybe_download_pretrained_vgg(data_dir="./data")

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


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """

    non_trivial_class = 2
    colors = np.array([[[0, 255, 0, 127]], [[255, 0, 0, 127]]])
    car_hood_mask = np.load("hood_mask.npy")
    car_hood_mask = skimage.transform.resize(car_hood_mask, image_shape).astype(np.bool)
    car_hood_mask = np.reshape(car_hood_mask, (*image_shape, 1))

    for image_file in glob(os.path.join(data_folder, 'CameraRGB', '*.png')):
        image = scipy.misc.imresize(skimage.io.imread(image_file), image_shape)

        raw_im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        street_im = scipy.misc.toimage(image)
        for color_index in range(non_trivial_class):
            im_softmax = raw_im_softmax[0][:, color_index + 1].reshape(image_shape[0], image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
            segmentation *= car_hood_mask
            mask = np.dot(segmentation, colors[color_index])
            mask = scipy.misc.toimage(mask, mode="RGBA")
            street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)


def save_inference_samples_2(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, data_dir, image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)


class ImageProcess(object):
    def __init__(self, sess, keep_prob, image_pl, logits, image_shape):
        self.sess = sess
        self.keep_prob = keep_prob
        self.image_pl = image_pl
        self.image_shape = image_shape
        self.logits = logits

        car_hood_mask = np.load("hood_mask.npy")
        car_hood_mask = skimage.transform.resize(car_hood_mask, self.image_shape).astype(np.bool)
        self.car_hood_mask = np.reshape(car_hood_mask, (*self.image_shape, 1))

    def pipeline(self, orig_image):
        image_shape = self.image_shape
        original_image_shape = orig_image.shape

        image, segmentation = self.classify(orig_image, 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        street_im = scipy.misc.imresize(street_im, original_image_shape[0:2])

        return np.array(street_im)

    def classify(self, orig_image, label):

        # TODO document what labels do
        image_shape = self.image_shape
        image = scipy.misc.imresize(orig_image, image_shape)
        im_softmax = self.sess.run(
            [tf.nn.softmax(self.logits)],
            {self.keep_prob: 1.0, self.image_pl: [image]})
        im_softmax = im_softmax[0][:, label].reshape(image_shape[0], image_shape[1])

        # TODO delete redundant reshape
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        segmentation *= self.car_hood_mask

        # TODO need to restore to original image shape

        return image, segmentation

    def get_encoded_sets(self, image_array):
        _, binary_road_result = self.classify(image_array, 1)
        _, binary_car_result = self.classify(image_array, 2)

        # The following code were used to debug the output of self.classify()
        #import pickle
        #pickle.dump(binary_road_result, open("temp.dump", 'wb'))

        # TODO magic 600x800 here
        binary_car_result = binary_car_result.astype(np.bool).reshape(*binary_car_result.shape[0:2])
        binary_road_result = binary_road_result.astype(np.bool).reshape(*binary_road_result.shape[0:2])

        binary_car_result = skimage.transform.resize(binary_car_result, (600, 800)).astype(np.uint8)
        binary_road_result = skimage.transform.resize(binary_road_result, (600, 800)).astype(np.uint8)

        return [encode(binary_car_result), encode(binary_road_result)]


# Define encoder function
def encode(array):
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")
