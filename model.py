import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import add

from ops import RAM, SubpixelConv2D, Normalization, Denormalization

class SRRAM():
    def __init__(self, scale_factor=2, channels=64, kernel_size=3, blocks=16):
        self.sf = scale_factor
        self.ch = channels
        self.k_size = kernel_size
        self.blocks = blocks

        self._build_model()

    def _build_model(self):
        input_image = Input(shape=(None, None, 3))
        x = Normalization()(input_image)

        # feature extraction part
        with tf.variable_scope('initial'):
            x = f0 = Conv2D(64, kernel_size=self.k_size, strides=1, padding='same')(x)
        # stacking RAM blocks
        for i in range(self.blocks):
            with tf.variable_scope('RAM_{}'.format(i)):
                x = RAM(x, channels=64)
        # adjust features
        with tf.variable_scope('initial'):
            x = Conv2D(64, kernel_size=self.k_size, strides=1, padding='same')(x)

        # global skip-connection
        x = add([x, f0])

        # upscale part
        if self.sf in [2, 3]:
            with tf.variable_scope('sub_pixel'):
                x = SubpixelConv2D(x.shape, scale=self.sf)(x)
        elif self.sf == 4:
            with tf.variable_scope('sub_pixel_1'):
                x = SubpixelConv2D(x.shape, scale=2)(x)
            with tf.variable_scope('sub_pixel_2'):
                x = SubpixelConv2D(x.shape, scale=2)(x)
        else:
            raise NotImplementedError

        # final convolution
        with tf.variable_scope('output'):
            x = Conv2D(3, kernel_size=self.k_size, strides=1, padding='same')(x)

        output_image = Denormalization()(x)

        #tf.summary.image('input_image', input_image)
        #tf.summary.image('output_image', output_image)

        self.model = Model(inputs=input_image, outputs=output_image)
