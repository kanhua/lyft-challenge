import sys

sys.path.append('/Users/kanhua/Dropbox/github/models/research/slim')

import tensorflow as tf

import math
import numpy as np
import tensorflow as tf
import time
import PIL
from datasets import imagenet

# Main slim library
from tensorflow.contrib import slim

from nets import mobilenet_v1


def mobilenetv1_fcn8(num_classes=3):
    layer4_out_name = 'MobilenetV1/MobilenetV1/Conv2d_4_pointwise/Relu6:0'
    layer6_out_name = 'MobilenetV1/MobilenetV1/Conv2d_6_pointwise/Relu6:0'
    layer13_out_name = 'MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu6:0'

    model_fname = "/Users/kanhua/Dropbox/Programming/tensorflow-math/mobilenet_v1_1.0_224_ckpt/mobilenet_v1_1.0_224_frozen.pb"
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
        with slim.arg_scope([slim.conv2d_transpose, slim.conv2d], padding='same', num_outputs=num_classes,
                            weights_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                            weights_initializer=tf.random_normal_initializer(stddev=0.01)):
            layer_13_convt = slim.conv2d_transpose(layer_13, kernel_size=(4, 4), stride=(2, 2))
            layer_6_conv = slim.conv2d(layer_6, kernel_size=(1, 1), stride=(1, 1))

            layer_seed_1 = tf.add(layer_13_convt, layer_6_conv)

            layer_seed_1_convt = slim.conv2d_transpose(layer_seed_1, kernel_size=(4, 4),
                                                       stride=(2, 2))

            layer_4_conv = slim.conv2d(layer_4, kernel_size=(1, 1), stride=(1, 1))

            layer_seed_2 = tf.add(layer_seed_1_convt, layer_4_conv)

            layer_logits = slim.conv2d_transpose(layer_seed_2, kernel_size=(16, 16),
                                                 stride=(8, 8))

    return image_pl, layer_logits, detect_graph


def show_graph(tf_graph):
    # with tf.Session(graph=detect_graph) as sess:

    logdir = "mobilenet_fcn_log/"
    with tf.Session(graph=tf_graph) as sess:
        writer = tf.summary.FileWriter("./" + logdir, sess.graph)

        writer.close()

    #import subprocess
    #subprocess.run(["tensorboard", "--logdir=$(pwd)" + logdir])


if __name__ == "__main__":
    image_pl, layer_logits, detect_graph = mobilenetv1_fcn8(num_classes=3)

    show_graph(tf_graph=detect_graph)
