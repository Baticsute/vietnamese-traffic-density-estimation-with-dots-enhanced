import numpy as np
from tensorflow import keras
import os

from skimage.io import imread
from skimage.transform import resize

# Learned from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class DatasetGenerator(keras.utils.Sequence):
    def __init__(
            self,
            image_files_ids,
            image_files_path,
            mask_files_path,
            mask_extension='.png',
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
        self.image_files_path = image_files_path
        self.mask_files_path = mask_files_path
        self.mask_extension = mask_extension
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

        # Generate data
        for i, id_ in enumerate(ids):
            # Read image files iteratively
            img = imread(self.image_files_path + id_)[:, :, :self.n_channels]
            img = resize(img, (self.img_h, self.img_w), mode='constant', preserve_range=True)
            images[i] = self.preprocess_img(img)

            mask_file_name = os.path.splitext(id_)[0] + '.png'
            mask = np.zeros((self.img_h, self.img_w))
            mask_ = imread(self.mask_files_path + mask_file_name, as_gray=True)
            # original size div to scale size
            mask = self.mapping_rescale_dot(mask, mask_)
            masks[i] = mask.reshape((self.img_h, self.img_w, 1))

        return images, masks

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.image_files_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def preprocess_img(self, img):
        """
        Preprocessing for the image
        z-score normalize
        """
        # return (img - img.mean()) / img.std()
        return img / 255.0

    def preprocess_label(self, mask):
        """
        Predict whole tumor. If you want to predict tumor sections, then
        just comment this out.
        """
        mask[mask > 0] = 1.0

        return mask

    def mapping_rescale_dot(self, mask_scale, mask_original):
        scale_factor_h = mask_scale.shape[0] / mask_original.shape[0]
        scale_factor_w = mask_scale.shape[1] / mask_original.shape[1]
        non_zero_points = np.array(np.nonzero(mask_original))
        non_zero_points[0] = non_zero_points[0] * scale_factor_h
        non_zero_points[1] = non_zero_points[1] * scale_factor_w
        non_zero_points = np.transpose(non_zero_points)
        for point in non_zero_points:
            x = point[0]
            y = point[1]
            mask_scale[x][y] = 1.0

        return mask_scale
