import sys

sys.path.append('/Users/kanhua/Dropbox/github/models/research/slim')

import math
import numpy as np
import tensorflow as tf

# Main slim library
from tensorflow.contrib import slim
from mobilenet_v1 import mobilenet_v1, mobilenet_v1_arg_scope
import vgg
import vgg_preprocessing


def linear_activation(x):
    return x


def mobilenet_rescale_from_uint8(images):
    images = tf.divide(images, 128)
    images = tf.subtract(images, 1)
    return images


def mobilenet_rescale_from_float(images):
    images = tf.multiply(images, 2)
    images = tf.subtract(images, 1)
    return images


def mobilenetv1_fcn8_model(images, num_classes, is_training=False, raw_image_shape=(520 - 170, 800),
                           decoder='fcn8'):
    train_image_shape = (224 * 2, 224 * 3)

    if decoder == 'fcn8':
        decoder_fn = mobilenet_v1_fcn_decoder
    elif decoder == 'fcn8_upsample':
        decoder_fn = mobilenet_v1_fcn8_upsample_decoder
    else:
        raise ValueError("the decoder should be either fcn8 or fcn8_upsample")
    # raw_image_shape=tf.constant((images.shape[2]),dtype=tf.int32)

    # images=tf.map_fn(lambda img: preprocess_image(img,224,224,is_training), images)

    images = rescale_and_resize_images(images, train_image_shape)

    with tf.contrib.slim.arg_scope(mobilenet_v1_arg_scope(is_training=is_training)):
        m_logits, end_points = mobilenet_v1(images, num_classes=1001,
                                            spatial_squeeze=False)

    layer4 = end_points['Conv2d_4_pointwise']
    layer6 = end_points['Conv2d_6_pointwise']
    layer13 = end_points['Conv2d_13_pointwise']

    last_layer = decoder_fn(layer13, layer4, layer6, num_classes)

    last_layer = post_process_logits(end_points, last_layer, raw_image_shape, train_image_shape)

    return last_layer, end_points


def vgg16_fcn8_model(images, num_classes, is_training=False, raw_image_shape=(520 - 170, 800),
                     decoder='fcn8'):
    train_image_shape = (224 * 2, 224 * 3)

    if decoder == 'fcn8':
        decoder_fn = mobilenet_v1_fcn_decoder
    elif decoder == 'fcn8_upsample':
        decoder_fn = mobilenet_v1_fcn8_upsample_decoder
    else:
        raise ValueError("the decoder should be either fcn8 or fcn8_upsample")

    if images.dtype != tf.uint8:
        raise ValueError("the image should be uint8")

    images = tf.image.resize_images(images, size=train_image_shape)
    tf.summary.image('input_image_after_rescale_and_resize',
                     tf.expand_dims(images[0], 0))

    processed_images = tf.map_fn(vgg_preprocessing.vgg_image_rescale,
                                 images,dtype=tf.float32)

    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(vgg.vgg_arg_scope()):
        # 1000 classes instead of 1001.
        logits, end_points = vgg.vgg_16(processed_images, num_classes=1000,
                                        is_training=is_training, spatial_squeeze=False)

    layer4 = end_points['vgg_16/pool3']
    layer6 = end_points['vgg_16/pool4']
    layer13 = end_points['vgg_16/pool5']

    last_layer = decoder_fn(layer13, layer4, layer6, num_classes)

    last_layer = post_process_logits(end_points, last_layer, raw_image_shape, train_image_shape)

    return last_layer, end_points


def rescale_and_resize_images(images, train_image_shape):
    tf.summary.image('input_image_before_rescale',
                     tf.expand_dims(images[0], 0))
    if images.dtype == tf.uint8:
        images = tf.cast(images, dtype=tf.float32)
        images = mobilenet_rescale_from_uint8(images)
    elif images.dtype == tf.float32:
        images = mobilenet_rescale_from_float(images)
    else:
        raise ValueError("the type of input image should be either uint8 or float32")
    # tf.summary.scalar("rescaled_image_sum",tf.reduce_sum(images[0],(0,1,2))/(600*800))
    if True:
        images = tf.image.resize_images(images, size=train_image_shape)
        tf.summary.image('input_image_after_rescale_and_resize',
                         tf.expand_dims(images[0], 0))
    return images


def post_process_logits(end_points, last_layer, raw_image_shape, train_image_shape):
    if raw_image_shape != tf.constant(train_image_shape):
        last_layer = tf.image.resize_images(last_layer, size=raw_image_shape)
    with tf.variable_scope("post_processing"):
        im_softmax = tf.nn.softmax(last_layer)

        # resize softmax if the the raw image size does not match the training image sizes
        # TODO this is redundant
        if raw_image_shape != tf.constant(train_image_shape):
            resized_im_softmax = tf.image.resize_images(im_softmax, size=raw_image_shape)
        else:
            resized_im_softmax = im_softmax
        end_points['im_softmax_zero'] = im_softmax[:, :, :, 0]
        end_points['im_softmax_road'] = im_softmax[:, :, :, 1]
        end_points['im_softmax_car'] = im_softmax[:, :, :, 2]
        tf.summary.image('output_softmax_before_rescale',
                         tf.expand_dims(tf.expand_dims(im_softmax[0, :, :, 0], 0), 3))

        end_points['resized_softmax_zero'] = resized_im_softmax[:, :, :, 0]
        end_points['resized_softmax_road'] = resized_im_softmax[:, :, :, 1]
        end_points['resized_softmax_car'] = resized_im_softmax[:, :, :, 2]
        tf.summary.image('output_softmax_after_rescale',
                         tf.expand_dims(tf.expand_dims(resized_im_softmax[0, :, :, 0], 0), 3))
        tf.summary.image('car_softmax_after_rescale',
                         tf.expand_dims(tf.expand_dims(resized_im_softmax[0, :, :, 2], 0), 3))
        tf.summary.image('road_softmax_after_rescale',
                         tf.expand_dims(tf.expand_dims(resized_im_softmax[0, :, :, 1], 0), 3))
    return last_layer


def mobilenetv1_fcn8(num_classes=3):
    layer4_out_name = 'MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6:0'
    layer6_out_name = 'MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6:0'
    layer13_out_name = 'MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6:0'

    model_fname = "./pretrained_models/mobilenet_v1_1.0_224_ckpt/mobilenet_v1_1.0_224_frozen.pb"
    # detect_graph = tf.Graph()
    detect_graph = tf.get_default_graph()
    with detect_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_fname, "rb") as fid:
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name="")
            image_pl = detect_graph.get_tensor_by_name('input:0')
            predictions_label = detect_graph.get_tensor_by_name('MobilenetV1/Predictions/Reshape_1:0')
            layer_13 = detect_graph.get_tensor_by_name(layer13_out_name)
            layer_4 = detect_graph.get_tensor_by_name(layer4_out_name)
            layer_6 = detect_graph.get_tensor_by_name(layer6_out_name)

            # double_x=tf.subtract(x,x)

    with detect_graph.as_default():
        layer_logits = mobilenet_v1_fcn_decoder(layer_13, layer_4, layer_6, num_classes)

    return image_pl, layer_logits, detect_graph


def mobilenet_v1_fcn_decoder(layer_13, layer_4, layer_6, num_classes):
    with slim.arg_scope([slim.conv2d_transpose, slim.conv2d], padding='same', num_outputs=num_classes,
                        weights_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                        weights_initializer=tf.random_normal_initializer(stddev=0.01),
                        activation_fn=linear_activation):
        layer_13_convt = slim.conv2d_transpose(layer_13, kernel_size=(4, 4), stride=(2, 2))
        layer_6_conv = slim.conv2d(layer_6, kernel_size=(1, 1), stride=(1, 1))

        layer_seed_1 = tf.add(layer_13_convt, layer_6_conv)

        layer_seed_1_convt = slim.conv2d_transpose(layer_seed_1, kernel_size=(4, 4),
                                                   stride=(2, 2))

        layer_4_conv = slim.conv2d(layer_4, kernel_size=(1, 1), stride=(1, 1))

        layer_seed_2 = tf.add(layer_seed_1_convt, layer_4_conv)

        layer_logits = slim.conv2d_transpose(layer_seed_2, kernel_size=(16, 16),
                                             stride=(8, 8))
    return layer_logits


def mobilenet_v1_fcn8_upsample_decoder(layer_13, layer_4, layer_6, num_classes):
    with slim.arg_scope([slim.conv2d_transpose, slim.conv2d], padding='same', num_outputs=num_classes,
                        weights_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                        weights_initializer=tf.random_normal_initializer(stddev=0.01),
                        activation_fn=linear_activation, stride=(1, 1)):
        shape = layer_13.shape.as_list()[1:3]
        l13_up = tf.image.resize_images(layer_13, size=(2 * shape[0], 2 * shape[1]))
        l13_up = slim.conv2d(l13_up, kernel_size=(1, 1), stride=(1, 1))
        l6_up = slim.conv2d(layer_6, kernel_size=(1, 1), stride=(1, 1))
        up_score_1 = tf.add(l13_up, l6_up)

        shape = layer_6.shape.as_list()[1:3]
        up_score_2 = tf.image.resize_images(up_score_1, size=(shape[0] * 2, shape[1] * 2))
        up_score_2 = slim.conv2d(up_score_2, kernel_size=(1, 1))
        l4_up = slim.conv2d(layer_4, kernel_size=(1, 1))
        up_score_2 = tf.add(l4_up, up_score_2)

        shape = up_score_2.shape.as_list()[1:3]
        up_score_3 = tf.image.resize_images(up_score_2, size=(shape[0] * 8, shape[1] * 8))
        layer_logits = slim.conv2d(up_score_3, kernel_size=(1, 1))

        return layer_logits


def show_graph(tf_graph):
    # with tf.Session(graph=detect_graph) as sess:

    logdir = "mobilenet_fcn_log/"
    with tf.Session(graph=tf_graph) as sess:
        writer = tf.summary.FileWriter("./" + logdir, sess.graph)

        writer.close()

    # import subprocess
    # subprocess.run(["tensorboard", "--logdir=$(pwd)" + logdir])


if __name__ == "__main__":
    image_pl, layer_logits, detect_graph = mobilenetv1_fcn8(num_classes=3)

    show_graph(tf_graph=detect_graph)
