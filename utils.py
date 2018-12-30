import tensorflow as tf
from tensorflow import keras

import os
from pathlib import Path

import datetime

def lr_decay(epoch, init_value=1e-4, decay_step=2e+5, decay_rate=2):
    x = init_value
    if x / decay_step < 1.:
        return x
    else:
        for _ in range(x // decay_step):
            x /= decay_rate
        return x

def build_save_path(flags):
    if not os.path.exists(flags.save_dir):
        os.mkdir(flags.save_dir)

    dataset = Path(flags.data_dir).stem
    size = '{}x{}'.format(flags.img_size, flags.img_size)
    srf = 'srf{}'.format(flags.scale_factor)
    batch_size = 'bs{}'.format(flags.batch_size)
    lr = 'lr{}'.format(flags.lr)
    save_path = os.path.join(flags.save_dir, '{}_{}_{}_{}_{}'.format(dataset, size, srf, batch_size, lr))
    
    if os.path.exists(save_path):
        return save_path + '_{}'.format(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    else:
        return save_path