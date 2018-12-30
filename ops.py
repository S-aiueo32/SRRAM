import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, ReLU
from tensorflow.keras.layers import Lambda, Activation
from tensorflow.keras.layers import add, multiply

import numpy as np
General100_RGB_MEAN = np.array([140.9, 127.1, 103.2])

def RAM(input, channels, kernel_size=3):
    # pre-attention feature extraction
    x = Conv2D(channels, kernel_size, strides=1, padding='same')(input)
    x = ReLU()(x)
    x = Conv2D(channels, kernel_size, strides=1, padding='same')(x)
    
    # compute attentions
    ca = CA(x, channels)
    sa = SA(x)
    fa = add([ca, sa])
    fa = Activation('sigmoid')(fa)

    # apply attention
    x = multiply([x, fa])

    return add([input, x])

def CA(input, channels, reduction_ratio=16):
    _, x = Lambda(lambda x: tf.nn.moments(x, axes=[1, 2]))(input)
    x = Dense(channels // reduction_ratio)(x)
    x = Dense(channels)(x)
    return x

def SA(input, kernel_size=3):
    x = DepthwiseConv2D(kernel_size, padding='same')(input)
    return x

def upsampler(input, scale, channels=64, kernel_size=3):
    x = Conv2D(channels * (scale ** 2), kernel_size, strides=1, padding='same')(input)
    x = SubpixelConv2D(x.shape, scale=scale)(x)
    return x

def SubpixelConv2D(input_shape, scale=4):
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape)

def Normalization(rgb_mean=General100_RGB_MEAN, **kwargs):
    return Lambda(lambda x: (x - rgb_mean) / 127.5, **kwargs)


def Denormalization(rgb_mean=General100_RGB_MEAN, **kwargs):
    return Lambda(lambda x: x * 127.5 + rgb_mean, **kwargs)