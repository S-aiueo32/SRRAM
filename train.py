import tensorflow as tf
from tensorflow.keras.callbacks import (TensorBoard, LearningRateScheduler,
                                        EarlyStopping, ModelCheckpoint)
from tensorflow.keras.optimizers import Adam

import numpy as np
from scipy.misc import imread, imsave, imresize

from model import SRRAM
from dataset import Dataset
import utils

from pathlib import Path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/General-100')
parser.add_argument('--save_dir', type=str, default='./saved')
parser.add_argument('--img_size', type=int, default=48)
parser.add_argument('--scale_factor', type=int, default=2)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--extension', type=str, default='bmp')
flags = parser.parse_args()

dataset = Dataset(flags)
srram = SRRAM(scale_factor=flags.scale_factor)

save_path = utils.build_save_path(flags)
cbs = [TensorBoard(log_dir=save_path, histogram_freq=1, write_graph=True),
       LearningRateScheduler(lambda epoch: utils.lr_decay(epoch, init_value=flags.lr)),
       EarlyStopping(monitor='val_loss', patience=1000, verbose=0, mode='auto'),
       ModelCheckpoint(save_path + '/model.e{epoch:05d}-loss{val_loss:.2f}.h5', save_best_only=True, period=100)]

srram.model.compile(optimizer=Adam(lr=flags.lr, epsilon=1e-8), loss='mae')
srram.model.fit(dataset.train_set, epochs=flags.epochs, steps_per_epoch=dataset.train_steps_per_epoch,
                validation_data=dataset.val_set, validation_steps=dataset.val_steps_per_epoch,
                callbacks=cbs)
#srram.model.save(Path(save_path) / 'model.h5')

sample_dir = Path(save_path) / 'sample'
sample_dir.mkdir(exist_ok=True)
for filename in (Path(flags.data_dir) / 'test').glob('*.bmp'):
    img = imread(filename)
    img = imresize(img, (img.shape[0] // flags.scale_factor, img.shape[1] // flags.scale_factor), interp='bicubic')
    out = np.squeeze(srram.model.predict(img[None, :, :, :]), axis=0)
    out = np.clip(out, 0, 255).astype('uint8')
    imsave(Path(sample_dir) / '{}.bmp'.format(filename.stem), out)
