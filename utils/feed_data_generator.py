import numpy as np
from tensorflow import keras
import os

from skimage.io import imread
import pathlib
import os
# Learned from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly


ROOT_PATH = str(pathlib.Path().absolute())
DATA_STORAGE_PATH = '/data_storage/'
STORAGE_PATH = ROOT_PATH + DATA_STORAGE_PATH

class DatasetGenerator(keras.utils.Sequence):
    def __init__(
            self,
            image_files_ids,
            dataset_name,
            data_type='train',
            file_extension='.npy',
            batch_size=32,
            img_h=192,
            img_w=256,
            n_channels=1,
            shuffle=True
    ):
        self.image_files_ids = image_files_ids
        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.dataset_name = dataset_name
        self.data_type = data_type
        self.file_extension = file_extension
        self.shuffle = shuffle
        self.n_channels = n_channels

        # self.image_files_ids = next(os.walk(image_files_path))[2]
        self.on_epoch_end()

    def __len__(self):
        # returns the number of batches
        return int(
            np.floor(len(self.image_files_ids) / self.batch_size)
        )

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        image_files_ids_temp = [self.image_files_ids[k] for k in indexes]
        X, y = self.__data_generation(image_files_ids_temp)

        return X, y

    def __data_generation(self, ids):
        # Initialization
        images = np.zeros((self.batch_size, self.img_h, self.img_w, self.n_channels), dtype=np.float32)
        masks = np.zeros((self.batch_size, self.img_h, self.img_w, 1), dtype=np.float32)
        base_path = STORAGE_PATH + self.dataset_name + '/' + self.data_type
        images_storage_path = base_path + '/images/'
        masks_storage_path = base_path + '/masks/'

        # Generate data
        for i, id_ in enumerate(ids):
            # Read image files iteratively

            file_name = os.path.splitext(id_)[0] + '.npy'
            img = np.load(images_storage_path + file_name)
            msk = np.load(masks_storage_path + file_name)

            images[i] = img
            masks[i] = msk

        return images, masks

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.image_files_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)
