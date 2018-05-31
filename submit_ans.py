import skvideo.io
import numpy as np
from main import build_eval_graph
import tensorflow as tf
from helper import encode
import scipy.misc
import sys
from simdata import UPPER_CUT



model_path = "./model_ckpt/model"

file = sys.argv[-1]
batch_size = 100
video = skvideo.io.vread(file)
num_classes = 3
write_video=False

if write_video:
    writer = skvideo.io.FFmpegWriter("outputvideo.mp4")


def paste_mask(rgb_frame, result_binary):
    street_im = scipy.misc.toimage(rgb_frame)
    mask = np.dot(np.expand_dims(result_binary, 2), [[255, 0, 0, 127]])
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im.paste(mask, box=None, mask=mask)
    return street_im


def infer_images(sess, input_image_pl,image_pad_pl, top_image_pad_pl,
                 expanded_rgb_frame, image_pad,softmax_car, softmax_road, merged_summary,top_image_pad):
    summary, result_car_image, result_road_image = sess.run([merged_summary, softmax_car, softmax_road],
                                                            feed_dict={input_image_pl: expanded_rgb_frame,
                                                                       image_pad_pl:image_pad,
                                                                       top_image_pad_pl:top_image_pad})
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

import tensorflow.contrib.slim as slim
with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    input_image, image_pad_pl, softmax_car, softmax_road, top_image_pad_pl = build_eval_graph()
    #saver=tf.train.Saver()
    #saver.restore(sess,model_path)
    get_var=slim.get_variables()
    init_fn=slim.assign_from_checkpoint_fn(model_path,get_var)
    init_fn(sess)

    train_writer = tf.summary.FileWriter('./log' + '/test', sess.graph)

    count = 0

    frame_batch = []
    batch_count = 0


    fe=FrameEncoder()

    merged = tf.summary.merge_all()

    for rgb_frame_id, rgb_frame in enumerate(video):
        # print(rgb_frame.max())
        frame_batch.append(rgb_frame)
        batch_count += 1
        if batch_count >= batch_size or (rgb_frame_id==(video.shape[0]-1) and batch_count >0):

            assert len(frame_batch)>0

            expanded_rgb_frame = np.array(frame_batch)
            expanded_rgb_frame=expanded_rgb_frame[:,:,:,:]
            image_pad = np.zeros((expanded_rgb_frame.shape[0],600 - 520, 800))
            top_image_pad= np.zeros((expanded_rgb_frame.shape[0],UPPER_CUT, 800))

            results = infer_images(sess, input_image,image_pad_pl,
                                   top_image_pad_pl, expanded_rgb_frame,
                                   image_pad,softmax_car, softmax_road, merged,top_image_pad)

            result_road_binary = results['road_binary']
            result_car_binary = results['car_binary']
            for idx, frame_in_batch in enumerate(frame_batch):
                if write_video:
                    street_im = paste_mask(frame_in_batch, result_car_binary[idx])
                    writer.writeFrame(street_im)
                fe.add_answer(result_car_binary[idx],result_road_binary[idx])

            train_writer.add_summary(results['summary'],rgb_frame_id)


            batch_count = 0
            frame_batch = []

import json
print(json.dumps(fe.answer_key))
