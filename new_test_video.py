import skvideo.io
import numpy as np
from main import build_eval_graph
import tensorflow as tf
from helper import encode
import scipy.misc


model_path = "./model_ckpt_kept/model"

merged = tf.summary.merge_all()

saver = tf.train.Saver()

file = "./Example/test_video.mp4"

video = skvideo.io.vread(file)

num_classes = 3

writer = skvideo.io.FFmpegWriter("outputvideo.mp4")

video = skvideo.io.vread(file)

input_image, softmax_car, softmax_road = build_eval_graph()


def paste_mask(rgb_frame, result_binary):
    street_im = scipy.misc.toimage(rgb_frame)
    mask = np.dot(np.expand_dims(result_binary, 2), [[255, 0, 0, 127]])
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im.paste(mask, box=None, mask=mask)
    return street_im


def infer_images(sess, expanded_rgb_frame, softmax_car, softmax_road, merged_summary):
    summary, result_car_image, result_road_image = sess.run([merged_summary, softmax_car, softmax_road],
                                                            feed_dict={input_image: expanded_rgb_frame})
    result_car_binary = (result_car_image > 0.5).astype(np.uint8)
    result_road_binary = (result_road_image > 0.5).astype(np.uint8)

    results = {}
    results['car_binary'] = result_car_binary
    results['road_binary'] = result_road_binary
    results['summary'] = summary

    return results


class FrameEncoder(object):

    def __init__(self):
        self.frame_count = 1
        self.answer_key = {}

    def add_answer(self, car, road):
        ans = [encode(car), encode(road)]

        self.answer_key[self.frame_count] = ans

        self.frame_count += 1


with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)
    train_writer = tf.summary.FileWriter('./log' + '/test', sess.graph)

    count = 0

    frame_batch = []
    batch_count = 0
    batch_size = 5

    fe=FrameEncoder()

    for rgb_frame_id, rgb_frame in enumerate(video):
        # print(rgb_frame.max())
        frame_batch.append(rgb_frame)
        batch_count += 1
        if batch_count >= batch_size or rgb_frame_id==(video.shape[0]-1):

            expanded_rgb_frame = np.array(frame_batch)

            results = infer_images(sess, expanded_rgb_frame, softmax_car, softmax_road, merged)

            result_road_binary = results['road_binary']
            result_car_binary = results['car_binary']
            for idx, frame_in_batch in enumerate(frame_batch):
                street_im = paste_mask(frame_in_batch, result_road_binary[idx])
                writer.writeFrame(street_im)
                fe.add_answer(result_car_binary[idx],result_road_binary[idx])


            batch_count = 0
            frame_batch = []

import json
print(json.dumps(fe.answer_key))
