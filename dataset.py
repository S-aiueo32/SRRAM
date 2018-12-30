import numpy as np
import tensorflow as tf
from tensorflow.image import ResizeMethod

from math import ceil
from glob import glob
from os.path import join

class Dataset():
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.img_size = config.img_size
        self.batch_size = config.batch_size
        self.sf = config.scale_factor
        self.extension = config.extension

        self._build_pipline()

    def _build_pipline(self):
        self.train_files = np.array(glob(join(self.data_dir, 'train/*.{}'.format(self.extension))))
        self.val_files = np.array(glob(join(self.data_dir, 'val/*.{}'.format(self.extension))))
        self.test_files = np.array(glob(join(self.data_dir, 'test/*.{}'.format(self.extension))))

        if len(self.train_files) != 0:
            train_set = tf.data.Dataset.from_tensor_slices(self.train_files)
            train_set = train_set.map(self._parse_fn)
            train_set = train_set.shuffle(1000)
            train_set = train_set.batch(self.batch_size)
            self.train_set = train_set.repeat()
            self.train_steps_per_epoch = ceil(len(self.train_files) / self.batch_size)

        if len(self.val_files) != 0:
            val_set = tf.data.Dataset.from_tensor_slices(self.val_files)
            val_set = val_set.map(self._parse_fn)
            val_set = val_set.batch(self.batch_size)
            self.val_set = val_set.repeat()
            self.val_steps_per_epoch = ceil(len(self.val_files) / self.batch_size)

        if len(self.test_files) != 0:
            test_set = tf.data.Dataset.from_tensor_slices(self.test_files)
            test_set = test_set.map(lambda x: self._parse_fn(x, augment=False))
            self.test_set = test_set.batch(1).repeat()

    def _parse_fn(self, filename, augment=True):
        if self.extension in ['jpg', 'jpeg', 'JPG', 'JPEG']:
            hr_image = tf.read_file(tf.cast(filename, tf.string))
            hr_image = tf.cast(tf.image.decode_jpeg(hr_image), tf.float32)
        elif self.extension in ['png', 'PNG']:
            hr_image = tf.read_file(tf.cast(filename, tf.string))
            hr_image = tf.cast(tf.image.decode_png(hr_image), tf.float32)
        elif self.extension in ['bmp', 'BMP']:
            hr_image = tf.read_file(tf.cast(filename, tf.string))
            hr_image = tf.cast(tf.image.decode_bmp(hr_image), tf.float32)
        elif self.extension in ['npz']:
            hr_image, *_ = tf.py_func(self._extract_npz, [filename], [tf.float32])
            hr_image.set_shape((None, None, 3))
        else:
            raise NotImplementedError

        if augment:
            hr_image = tf.random_crop(hr_image, size=(self.img_size, self.img_size, 3))
            hr_image = tf.image.random_flip_up_down(hr_image)
            hr_image = tf.image.random_flip_left_right(hr_image)
            hr_image = tf.image.rot90(hr_image, k=tf.random_uniform((), minval=0, maxval=4, dtype=tf.int32))
        else:
            hr_image = self._chop(hr_image)

        down_size = tf.cast(tf.shape(hr_image)[:-1]/2, tf.int32)
        lr_image = tf.image.resize_images(hr_image, size=down_size, method=ResizeMethod.BICUBIC)
        return lr_image, hr_image

    def _chop(self, image):
        size = tf.unstack(tf.shape(image)[:-1])
        return image[:size[0]-tf.mod(size[0], self.sf), :size[1]-tf.mod(size[1], self.sf), :]

    def _extract_npz(self, filename):
        with np.load(filename.decode('utf-8')) as f:
            return f['arr_0'].astype('float32')
