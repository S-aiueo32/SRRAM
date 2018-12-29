import os
import shutil
from glob import glob

import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='0B7tU5Pj1dfCMVVdJelZqV0prWnM',
                                    dest_path='./General100.zip',
                                    unzip=True)

os.mkdir('./General-100/train')
os.mkdir('./General-100/test')
os.mkdir('./General-100/val')

filenames = np.array(glob('./General-100/*.bmp'))
train_files = np.random.choice(filenames, size=80, replace=False)
for filename in train_files:
    shutil.move(filename, './General-100/train')

test_val_files = np.array(list(set(filenames) - set(train_files)))
test_files = np.random.choice(test_val_files, size=10, replace=False)
for filename in test_files:
    shutil.move(filename, './General-100/test')

val_files = np.array(list(set(test_val_files) - set(test_files)))
for filename in val_files:
    shutil.move(filename, './General-100/val')
