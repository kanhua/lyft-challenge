import os.path
import tensorflow as tf
import helper
import warnings
import numpy as np
from distutils.version import LooseVersion
import project_tests as tests

from simdata import ImageNpy

from mobilenet_v1_fcn8 import mobilenetv1_fcn8_model


def optimize(nn_last_layer, correct_label, learning_rate, num_classes, global_step):
    """
    Build the TensorFLow loss and optimizer operations.

    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss, global_step=global_step)

    return logits, train_op, cross_entropy_loss


def train_mobilenet_v1_fcn8(load_model="latest", shift_hue_prob=0):
    num_classes = 3
    image_shape = (224 * 2, 224 * 3)

    image_data = ImageNpy("./data/train_data.npy", "./data/train_label.npy")
    get_batches_fn = image_data.get_bathes_fn_with_crop

    # Load pretrained mobilenet_v1
    import tensorflow.contrib.slim as slim
    pretrained_model_path = "./pretrained_models/mobilenet_v1_1.0_224_ckpt/mobilenet_v1_1.0_224.ckpt"

    input_image = tf.placeholder(tf.uint8, shape=(None, None, None, 3))
    correct_label = tf.placeholder(tf.uint8, [None, None, None, num_classes], name='correct_label')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    stacked_image_label = tf.concat((input_image, correct_label), axis=3)

    from inception_preprocessing import crop_image_label_for_train, distort_color, random_distort_images, \
        preprocess_image_label

    cropped_stacked_image_label = tf.map_fn(
        lambda img: preprocess_image_label(img, cropped_shape=None),
        stacked_image_label, dtype=tf.uint8)

    cropped_input_image = cropped_stacked_image_label[:, :, :, 0:3]
    cropped_label = cropped_stacked_image_label[:, :, :, 3:3 + num_classes]

    cropped_input_image = tf.map_fn(random_distort_images, cropped_input_image, dtype=tf.float32)
    # cropped_input_image=tf.map_fn(lambda img: distort_color(img),cropped_input_image)

    tf.summary.image('cropped_label', tf.expand_dims(cropped_label[:, :, :, 1], axis=3))

    final_layer, endpoints = mobilenetv1_fcn8_model(cropped_input_image, num_classes=3, is_training=True)

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    logits, train_op, cross_entropy_loss = optimize(final_layer, cropped_label,
                                                    learning_rate, num_classes, global_step)

    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)

    merged = tf.summary.merge_all()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        if load_model == 'mobilenetv1':
            get_var = slim.get_model_variables('MobilenetV1')
            sess_load = slim.assign_from_checkpoint_fn(pretrained_model_path, get_var)
            sess_load(sess)
            sess.run(tf.global_variables_initializer())
        elif load_model == "latest":
            # saver.restore(sess,"./model_ckpt_udacity_trained/model")
            get_var = slim.get_variables()
            sess_load = slim.assign_from_checkpoint_fn("./model_ckpt/model", get_var)
            sess_load(sess)
            # sess.run(tf.global_variables_initializer())
        else:
            raise ValueError("model wrong!")

        # print(slim.get_model_variables())
        # print(len(slim.get_model_variables()))

        train_writer = tf.summary.FileWriter('./log' + '/train', sess.graph)

        epochs = 50
        batch_size = 20
        for ep in range(epochs):
            print("epoch: {}".format(ep))
            for image, label in get_batches_fn(batch_size, crop_size=image_shape, shift_hue_prob=shift_hue_prob,
                                               filter=True):
                summary, _, loss, step_count = sess.run([merged, train_op, cross_entropy_loss, global_step],
                                                        feed_dict={input_image: image, correct_label: label,
                                                                   learning_rate: 0.001})
                print("loss: = {:.5f}".format(loss))
                train_writer.add_summary(summary, global_step=step_count)
                saver.save(sess, './model_ckpt/model')


def eval_mobilenet_v1_fcn8(image):
    input_image, softmax_car, softmax_road = build_eval_graph()

    model_path = "./model_ckpt/model"

    merged = tf.summary.merge_all()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_path)
        train_writer = tf.summary.FileWriter('./log' + '/test', sess.graph)

        count = 0
        summary, result_car_image, result_road_image = sess.run([merged, softmax_car, softmax_road],
                                                                feed_dict={input_image: image})
        train_writer.add_summary(summary, count)
        count += 1

    result_car_binary = (result_car_image > 0.5)
    result_road_binary = (result_road_image > 0.5)

    return result_car_binary, result_road_binary


def mask_engine_hood(softmax_tensor):
    car_hood_mask = np.load("hood_mask.npy")

    mask = tf.constant(car_hood_mask, dtype=np.float32)

    masked_softmax = tf.map_fn(lambda x: tf.multiply(x, mask), softmax_tensor)

    return masked_softmax


def build_eval_graph():
    num_classes = 3
    train_image_shape = (224 * 2, 224 * 3)

    input_image = tf.placeholder(tf.uint8, shape=(None, None, None, 3))
    image_pad = tf.placeholder(tf.float32, shape=(None, None, None))
    crop_input_image = input_image[:, 0:520, :, :]

    from mobilenet_v1_fcn8 import mobilenet_rescale_from_uint8
    images = mobilenet_rescale_from_uint8(crop_input_image)
    images = tf.image.resize_images(images, size=train_image_shape)
    final_layer, endpoints = mobilenetv1_fcn8_model(images, num_classes=num_classes,
                                                    is_training=True, raw_image_shape=(520, 800))
    softmax_car = endpoints['resized_softmax_car']
    softmax_road = endpoints['resized_softmax_road']

    softmax_road = tf.concat((softmax_road, image_pad), 1)
    softmax_car = tf.concat((softmax_car, image_pad), 1)

    with tf.variable_scope("car_pred"):
        softmax_car = mask_engine_hood(softmax_car)
    with tf.variable_scope("road_pred"):
        softmax_road = mask_engine_hood(softmax_road)
    return input_image, image_pad, softmax_car, softmax_road


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
    train_mobilenet_v1_fcn8(load_model='latest', shift_hue_prob=0.5)
