import unittest
from main import eval_mobilenet_v1_fcn8
from simdata import ImageNpy
import skimage.io
import numpy as np
import skvideo.io


class MyTestCase(unittest.TestCase):
    def test_something(self):
        image_data = ImageNpy("./data/train_data.npy", "./data/train_label.npy")

        test_images = image_data.images[0:10, :, :, :]

        eval_mobilenet_v1_fcn8(test_images)

    def test_single_image(self):
        image_file = "./data/CameraRGB/5.png"
        img_array = skimage.io.imread(image_file)

        img_array = np.expand_dims(img_array, 0)

        eval_mobilenet_v1_fcn8(img_array)

    def test_video(self):
        file = "./Example/test_video.mp4"
        video = skvideo.io.vread(file)

        answer_key = {}

        # Frame numbering starts at 1
        frame = 1

        num_classes = 3
        import scipy.misc
        writer = skvideo.io.FFmpegWriter("outputvideo.mp4")
        for rgb_frame in video:
            print(rgb_frame.max())
            expanded_rgb_frame = np.expand_dims(rgb_frame, 0)
            eval_mobilenet_v1_fcn8(expanded_rgb_frame)

            #street_im = scipy.misc.toimage(rgb_frame)
            #mask = np.dot(np.expand_dims(x[0],2), [[255, 0, 0,127]])
            #mask = scipy.misc.toimage(mask, mode="RGBA")
            #street_im.paste(mask, box=None, mask=mask)

            writer.writeFrame(rgb_frame)


if __name__ == '__main__':
    unittest.main()
