import skvideo.io
import numpy as np
from main import build_eval_graph
import tensorflow as tf
from helper import encode

input_image, softmax_car, softmax_road = build_eval_graph()

model_path = "./model_ckpt_kept/model"

merged = tf.summary.merge_all()

saver = tf.train.Saver()

file = "./Example/test_video.mp4"

video = skvideo.io.vread(file)

answer_key = {}

# Frame numbering starts at 1
frame = 1

num_classes = 3
import scipy.misc

writer = skvideo.io.FFmpegWriter("outputvideo.mp4")

video = skvideo.io.vread(file)

car_hood_mask = np.load("hood_mask.npy")


def paste_mask(rgb_frame, result_binary):
    street_im = scipy.misc.toimage(rgb_frame)
    mask = np.dot(np.expand_dims(result_binary, 2), [[255, 0, 0, 127]])
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im.paste(mask, box=None, mask=mask)
    return street_im


with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)
    train_writer = tf.summary.FileWriter('./log' + '/test', sess.graph)

    count = 0

    frame_batch = []
    batch_count = 0
    batch_size = 5
    for rgb_frame in video:
        # print(rgb_frame.max())
        frame_batch.append(rgb_frame)
        batch_count += 1
        if batch_count >= batch_size:

            expanded_rgb_frame = np.array(frame_batch)

            summary, result_car_image, result_road_image = sess.run([merged, softmax_car, softmax_road],
                                                                    feed_dict={input_image: expanded_rgb_frame})

            result_car_binary = (result_car_image > 0.5).astype(np.uint8)
            result_road_binary = (result_road_image > 0.5).astype(np.uint8)


            for idx,frame_in_batch in enumerate(frame_batch):
                street_im = paste_mask(frame_in_batch, result_road_binary[idx]*car_hood_mask)
                writer.writeFrame(street_im)
                answer_key[frame]=[encode(result_car_binary[idx]*car_hood_mask),encode(result_road_binary[idx]*car_hood_mask)]
                frame+=1

            batch_count = 0
            frame_batch = []

    expanded_rgb_frame = np.array(frame_batch)

    summary, result_car_image, result_road_image = sess.run([merged, softmax_car, softmax_road],
                                                            feed_dict={input_image: expanded_rgb_frame})

    result_car_binary = (result_car_image > 0.5).astype(np.uint8)
    result_road_binary = (result_road_image > 0.5).astype(np.uint8)

    for idx, frame_in_batch in enumerate(frame_batch):
        street_im = paste_mask(frame_in_batch, result_road_binary[idx]*car_hood_mask)
        writer.writeFrame(street_im)
        answer_key[frame] = [encode(result_car_binary[idx]*car_hood_mask), encode(result_road_binary[idx]*car_hood_mask)]
        frame += 1

    batch_count = 0
    frame_batch = []

import json
print(json.dumps(answer_key))
