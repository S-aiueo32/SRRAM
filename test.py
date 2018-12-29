import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import add
from tensorflow.keras.callbacks import TensorBoard

import numpy as np
from scipy.misc import imread, imsave, imresize

from pathlib import Path

from model import SRRAM
from dataset import Dataset
import utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=True)
parser.add_argument('--filename', type=str, required=True)
flags = parser.parse_args()

model = load_model(str(Path(flags.model_dir) / 'model.h5'))

filename = flags.filename
scale_factor = int(Path(flags.model_dir).stem.split('_')[2].replace('srf', ''))
img = imread(filename)
img = imresize(img, (img.shape[0] // scale_factor, img.shape[1] // scale_factor), interp='bicubic')
out = np.squeeze(model.predict(img[None, :, :, :]), axis=0)
imsave(Path(filename).stem + '.bmp', out)
