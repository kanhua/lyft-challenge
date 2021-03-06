import os.path
import tensorflow as tf
import helper
import warnings
import numpy as np
from distutils.version import LooseVersion
import project_tests as tests

from simdata import ImageNpy

from mobilenet_v1_fcn8 import mobilenetv1_fcn8_model


def vgg_encoder(sess, vgg_path, num_classes):
    input_image, keep, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)

    final_output_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)

    return input_image, keep, final_output_layer


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer_3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer_4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer_7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_image, keep, layer_3_out, layer_4_out, layer_7_out


# tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    layer_7_conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01))

    layer_7_out = tf.layers.conv2d_transpose(layer_7_conv_1x1, num_classes, 4, 2, padding='same',
                                             kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    layers_4_conv_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    layers_4_out = tf.add(layers_4_conv_1x1, layer_7_out)

    layer_4_up = tf.layers.conv2d_transpose(layers_4_out, num_classes, 4,
                                            strides=2, padding='same')

    layer_3_conv_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    layer_3_add = tf.add(layer_4_up, layer_3_conv_1x1)

    final_output_layer = tf.layers.conv2d_transpose(layer_3_add,
                                                    num_classes, 16, strides=8,
                                                    padding='same',
                                                    kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return final_output_layer


# tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
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
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss


# tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label,
             keep_prob, learning_rate, saver=None):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param saver: tf.train.Saver object
    """

    sess.run(tf.global_variables_initializer())
    # print(tf.global_variables(scope=None))

    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter('./log' + '/train')

    count = 0
    for ep in range(epochs):
        print("epoch: {}".format(ep))
        for image, label in get_batches_fn(batch_size):
            image = image.astype(np.float32) / 128 - 1
            summary, _, loss = sess.run([merged, train_op, cross_entropy_loss],
                                        feed_dict={input_image: image, correct_label: label,  # keep_prob: 0.5,
                                                   learning_rate: 0.001})
            print("loss: = {:.5f}".format(loss))
            train_writer.add_summary(summary, count)
            count += 1
        if saver is not None:
            saver.save(sess, './model_ckpt/model')

    train_writer.close()


# tests.test_train_nn(train_nn)

def train_mobilenet_v1_fcn8(load_model="latest"):
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

    logits, train_op, cross_entropy_loss = optimize(final_layer, cropped_label, learning_rate, num_classes)

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

        print(slim.get_model_variables())
        print(len(slim.get_model_variables()))

        train_writer = tf.summary.FileWriter('./log' + '/train', sess.graph)

        count = 0
        epochs = 50
        batch_size = 20
        for ep in range(epochs):
            print("epoch: {}".format(ep))
            for image, label in get_batches_fn(batch_size, crop_size=image_shape):
                summary, _, loss = sess.run([merged, train_op, cross_entropy_loss],
                                            feed_dict={input_image: image, correct_label: label,
                                                       learning_rate: 0.001})
                print("loss: = {:.5f}".format(loss))
                train_writer.add_summary(summary, count)
                count += 1
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
    image_pad=tf.placeholder(tf.float32, shape=(None,None,None))
    crop_input_image = input_image[:, 0:520, :, :]

    from mobilenet_v1_fcn8 import mobilenet_rescale_from_uint8
    images = mobilenet_rescale_from_uint8(crop_input_image)
    images = tf.image.resize_images(images, size=train_image_shape)
    final_layer, endpoints = mobilenetv1_fcn8_model(images, num_classes=num_classes,
                                                    is_training=True, raw_image_shape=(520, 800))
    softmax_car = endpoints['resized_softmax_car']
    softmax_road = endpoints['resized_softmax_road']

    softmax_road = tf.concat((softmax_road,image_pad),1)
    softmax_car = tf.concat((softmax_car,image_pad),1)


    with tf.variable_scope("car_pred"):
        softmax_car = mask_engine_hood(softmax_car)
    with tf.variable_scope("road_pred"):
        softmax_road = mask_engine_hood(softmax_road)
    return input_image, image_pad,softmax_car, softmax_road


def run():
    num_classes = 3
    image_shape = (224, 224)
    # image_shape = (64, 64)
    vgg_dir = './data'

    data_dir = './data'
    val_data_dir = './data/'
    runs_dir = './runs'
    # tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    # helper.maybe_download_pretrained_vgg(vgg_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    image_data = ImageNpy("./data/train_data.npy", "./data/train_label.npy")
    get_batches_fn = image_data.get_batches_fn

    with tf.Session() as sess:
        # Path to vgg model
        # Create function to get batches
        # get_batches_fn = helper.gen_batch_function(data_dir, image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # correct_label_image=tf.placeholder(tf.int32,[None,image_shape[0],image_shape[1]],name="correct_label_image")
        # correct_label = tf.placeholder(correct_label_image, depth=num_classes, name='correct_label')
        import tensorflow.contrib.slim as slim
        model_path = "./pretrained_models/mobilenet_v1_1.0_224_ckpt/mobilenet_v1_1.0_224.ckpt"
        get_var = slim.get_model_variables('MobilenetV1')
        sess_load = slim.assign_from_checkpoint_fn(model_path, get_var)

        sess_load(sess)

        input_image, keep_prob, logits, train_op, cross_entropy_loss, correct_label, learning_rate = foward_pass_2(
            num_classes)

        # set up a saver object
        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state('./model_ckpt/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # Train NN using the train_nn function
        train_nn(sess, 50, 10, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob,
                 learning_rate, saver=saver)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples_2(runs_dir, val_data_dir, sess, image_shape, logits, keep_prob, input_image)


def foward_pass(num_classes, sess, vgg_dir):
    vgg_path = os.path.join(vgg_dir, 'vgg')
    correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    input_image, keep_prob, layer_output = vgg_encoder(sess, vgg_path, num_classes)
    logits, train_op, cross_entropy_loss = optimize(layer_output,
                                                    correct_label, learning_rate, num_classes)
    return input_image, keep_prob, logits, train_op, cross_entropy_loss, correct_label, learning_rate


def foward_pass_2(num_classes, is_training):
    correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    from mobilenet_v1_fcn8 import mobilenetv1_fcn8
    input_image, layer_output, _ = mobilenetv1_fcn8(num_classes=num_classes)
    keep_prob = None
    logits, train_op, cross_entropy_loss = optimize(layer_output,
                                                    correct_label, learning_rate, num_classes)
    return input_image, keep_prob, logits, train_op, cross_entropy_loss, correct_label, learning_rate


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
    train_mobilenet_v1_fcn8(load_model='latest')
