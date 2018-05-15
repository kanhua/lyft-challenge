import sys, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import skimage.io
import tensorflow as tf

from main import foward_pass
from helper import ImageProcess

file = sys.argv[-1]

if file == 'demo.py':
    print("Error loading video")
    quit


# Define encoder function
def encode(array):
    assert array.shape==(600,800)
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def naive_calssify(image_array):
    red = image_array[:, :, 0]
    # Look for red cars :)
    binary_car_result = np.where(red > 250, 1, 0).astype('uint8')

    # Look for road :)
    binary_road_result = binary_car_result = np.where(red < 20, 1, 0).astype('uint8')

    return [encode(binary_car_result), encode(binary_road_result)]


def test_output():
    vgg_dir = './data'
    num_classes = 3
    with tf.Session() as sess:
        input_image, keep_prob, logits, train_op, cross_entropy_loss, correct_label, learning_rate \
            = foward_pass(num_classes, sess, vgg_dir)

        saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state('./model_ckpt/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        ip = ImageProcess(sess, keep_prob, input_image, logits, image_shape=(640, 832))

        pip_image = ip.pipeline(skimage.io.imread("./data/CameraRGB/0.png"))

        import matplotlib.pyplot as plt
        plt.imshow(pip_image)
        plt.show()


if __name__ == "__main__":
    test_image = skimage.io.imread("./data/CameraRGB/0.PNG")

    naive_calssify(test_image)

    test_output()
