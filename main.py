from os.path import join
import tensorflow as tf
import helper
import warnings
import numpy as np
from distutils.version import LooseVersion
import project_tests as tests

from simdata import ImageNpy, CLASS_WEIGHT

from mobilenet_v1_fcn8 import mobilenetv1_fcn8_model,vgg16_fcn8_model

from simdata import UPPER_CUT
from inception_preprocessing import random_distort_images, preprocess_image_label
import tensorflow.contrib.slim as slim


def optimize(nn_last_layer, correct_label, learning_rate, global_step, add_class_weight=False):
    """
    Build the TensorFLow loss and optimizer operations.

    :param add_class_weight:
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    if add_class_weight:
        weight = tf.constant(CLASS_WEIGHT, dtype=tf.float32)
        f_correct_label = tf.cast(correct_label, dtype=tf.float32)
        weighted_label = tf.multiply(f_correct_label, weight)

        weighted_label = tf.reduce_sum(weighted_label, axis=3)
        # print(weighted_label.shape)

        r_correct_label = tf.reshape(correct_label, shape=(-1, 3))
        r_last_layer = tf.reshape(nn_last_layer, shape=(-1, 3))
        # r_weighted_label=tf.reshape(weighted_label,shape=(-1,3))

        cross_entropy_image = tf.losses.softmax_cross_entropy(onehot_labels=r_correct_label,
                                                              logits=r_last_layer)
        cross_entropy_image = cross_entropy_image * weighted_label
    else:

        r_correct_label = tf.reshape(correct_label, shape=(-1, 3))
        r_last_layer = tf.reshape(nn_last_layer, shape=(-1, 3))
        cross_entropy_image=tf.losses.softmax_cross_entropy(onehot_labels=r_correct_label,logits=r_last_layer)

        cross_entropy_loss = tf.reduce_mean(cross_entropy_image)
    # cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=nn_last_layer,
    #                                                                               labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss, global_step=global_step)

    return train_op, cross_entropy_loss


def train_mobilenet_v1_fcn8(load_model="latest", shift_hue_prob=0,
                            add_class_weight=False,batch_size=20,
                            set_learning_rate=1e-3,data_aug_faster_mode=False):
    """
    Main program for training

    :param load_model: "latest", "mobilenetv1" or "vgg16"
    :param shift_hue_prob: the portion of images that shifts the hue values in the vehicles
    :param add_class_weight: add weighting in the loss function to balance vehicles
    :param batch_size: number of data per batch
    :param set_learning_rate: learning rate
    :param data_aug_faster_mode: faster mode in inception_preprocessing.random_distort_images()
    :return:
    """
    num_classes = 3

    image_data = ImageNpy(join("data","train_data_baseline.npy"), join("data","train_label_baseline.npy"))
    get_batches_fn = image_data.get_bathes_fn_with_crop

    # Load pretrained mobilenet_v1
    pretrained_model_path = join("pretrained_models","mobilenet_v1_1.0_224_ckpt","mobilenet_v1_1.0_224.ckpt")

    input_image = tf.placeholder(tf.uint8, shape=(None, None, None, 3))
    correct_label = tf.placeholder(tf.uint8, [None, None, None, num_classes], name='correct_label')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    stacked_image_label = tf.concat((input_image, correct_label), axis=3)

    cropped_stacked_image_label = tf.map_fn(
        lambda img: preprocess_image_label(img, cropped_shape=None),
        stacked_image_label, dtype=tf.uint8)

    cropped_input_image = cropped_stacked_image_label[:, :, :, 0:3]
    cropped_label = cropped_stacked_image_label[:, :, :, 3:3 + num_classes]

    #tf.summary.image('cropped_label', tf.expand_dims(cropped_label[:, :, :, 1], axis=3))

    final_layer, endpoints = mobilenetv1_fcn8_model(cropped_input_image, num_classes=3, is_training=True,
                                                    raw_image_shape=(520 - UPPER_CUT, 800),
                                                    decoder="fcn8",data_aug_faster_mode=data_aug_faster_mode)

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    train_op, cross_entropy_loss = optimize(final_layer, cropped_label,
                                            learning_rate, global_step,add_class_weight=add_class_weight)

    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)

    merged = tf.summary.merge_all()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        if load_model == 'mobilenetv1':
            get_var = slim.get_model_variables('MobilenetV1')
            sess_load = slim.assign_from_checkpoint_fn(pretrained_model_path, get_var)
            sess_load(sess)
            sess.run(tf.global_variables_initializer())
        elif load_model=='vgg16':
            get_var = slim.get_model_variables('vgg_16')
            vgg_pretrained_path=join("pretrained_models","vgg16/vgg_16.ckpt")
            sess_load = slim.assign_from_checkpoint_fn(vgg_pretrained_path, get_var)
            sess_load(sess)
            sess.run(tf.global_variables_initializer())

        elif load_model == "latest":
            # saver.restore(sess,"./model_ckpt_udacity_trained/model")
            get_var = slim.get_variables()
            sess_load = slim.assign_from_checkpoint_fn(join("model_ckpt","model", get_var))
            sess_load(sess)
            # sess.run(tf.global_variables_initializer())
        else:
            raise ValueError("model wrong!")

        # print(slim.get_model_variables())
        # print(len(slim.get_model_variables()))

        train_writer = tf.summary.FileWriter(join('log' , 'train'), sess.graph)

        epochs = 15
        for ep in range(epochs):
            print("epoch: {}".format(ep))
            for image, label in get_batches_fn(batch_size, crop_size=None, shift_hue_prob=shift_hue_prob,
                                               filter=False):
                summary, _, loss, step_count = sess.run([merged, train_op, cross_entropy_loss, global_step],
                                                        feed_dict={input_image: image, correct_label: label,
                                                                   learning_rate: set_learning_rate})
                print("loss: = {:.5f}".format(loss))
                train_writer.add_summary(summary, global_step=step_count)
                saver.save(sess, join('model_ckpt','model'))


def mask_engine_hood(softmax_tensor):
    car_hood_mask = np.load("hood_mask.npy")

    mask = tf.constant(car_hood_mask, dtype=np.float32)

    masked_softmax = tf.map_fn(lambda x: tf.multiply(x, mask), softmax_tensor)

    return masked_softmax


def build_eval_graph():
    num_classes = 3

    input_image = tf.placeholder(tf.uint8, shape=(None, None, None, 3))
    image_pad = tf.placeholder(tf.float32, shape=(None, None, None))
    top_image_pad = tf.placeholder(tf.float32, shape=(None, None, None))
    crop_input_image = input_image[:, UPPER_CUT:520, :, :]

    final_layer, endpoints = mobilenetv1_fcn8_model(crop_input_image, num_classes=num_classes,
                                                    is_training=False, raw_image_shape=(520 - UPPER_CUT, 800),
                                                    decoder='fcn8')
    #data_num=input_image.get_shape().as_list()[0]
    #t_top_image_pad=tf.zeros((data_num,180,800))

    softmax_car = endpoints['resized_softmax_car']
    softmax_road = endpoints['resized_softmax_road']

    softmax_road = tf.concat((top_image_pad, softmax_road, image_pad), 1)
    softmax_car = tf.concat((top_image_pad, softmax_car, image_pad), 1)

    with tf.variable_scope("car_pred"):
        softmax_car = mask_engine_hood(softmax_car)
    with tf.variable_scope("road_pred"):
        softmax_road = mask_engine_hood(softmax_road)
    return input_image, image_pad, softmax_car, softmax_road, top_image_pad


if __name__ == '__main__':
    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion(
        '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    train_mobilenet_v1_fcn8(load_model='mobilenetv1', shift_hue_prob=0,
                            add_class_weight=False, batch_size=20,
                            set_learning_rate=1e-3,data_aug_faster_mode=False)
