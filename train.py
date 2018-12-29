import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import add
from tensorflow.keras.callbacks import TensorBoard

import numpy as np
from scipy.misc import imread, imsave, imresize

import os
from pathlib import Path
from glob import glob

from model import SRRAM
from dataset import Dataset
import utils


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/General-100')
parser.add_argument('--save_dir', type=str, default='./saved')
parser.add_argument('--img_size', type=int, default=48)
parser.add_argument('--scale_factor', type=int, default=2)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-8)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--extension', type=str, default='bmp')
flags = parser.parse_args()

dataset = Dataset(flags)
srram = SRRAM(scale_factor=2)

save_path = utils.build_save_path(flags)
tb = TensorBoard(log_dir=save_path, histogram_freq=1, write_graph=True)

srram.model.compile(optimizer=tf.train.AdamOptimizer(flags.lr, flags.beta1, flags.beta2), loss='mae', metrics=['mae', 'mse'])
srram.model.fit(dataset.train_set, epochs=flags.epochs, steps_per_epoch=dataset.train_steps_per_epoch,
                validation_data=dataset.val_set, validation_steps=dataset.val_steps_per_epoch,
                callbacks=[tb])
srram.model.save(str(Path(save_path) / 'model.h5'))

sample_dir = str(Path(save_path) / 'sample')
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)
for filename in glob(str(Path(flags.data_dir) / 'test' / '*.bmp')):
    img = imread(filename)
    img = imresize(img, (img.shape[0] // flags.scale_factor, img.shape[1] // flags.scale_factor), interp='bicubic')
    out = np.squeeze(srram.model.predict(img[None, :, :, :]), axis=0)
    imsave(str(Path(sample_dir) / Path(filename).stem) + '.bmp', out)


