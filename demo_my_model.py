import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import tensorflow as tf
import skimage.io
from main import foward_pass,foward_pass_2
from helper import ImageProcess, encode
import warnings

file = sys.argv[-1]


# Define encoder function
def encode(array):
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")


video = skvideo.io.vread(file)

answer_key = {}

# Frame numbering starts at 1
frame = 1

vgg_dir = './data'
num_classes = 3
with tf.Session() as sess:
    input_image, keep_prob, logits, train_op, cross_entropy_loss, correct_label, learning_rate = foward_pass_2(
        num_classes)

    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state('./model_ckpt/')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    ip = ImageProcess(sess, keep_prob, input_image, logits, image_shape=(224, 224))

    writer = skvideo.io.FFmpegWriter("outputvideo.mp4")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rgb_frame in video:
            pip_image = ip.pipeline(rgb_frame)

            writer.writeFrame(pip_image)

            encoded_frame = ip.get_encoded_sets(rgb_frame)

            answer_key[frame] = encoded_frame
            frame += 1

print(json.dumps(answer_key))
