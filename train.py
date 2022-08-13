from utils import data_loader
from utils import model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import tensorflow as tf

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
tf.keras.backend.set_image_data_format('channels_last')  # TF dimension ordering in this code

train_data = data_loader.load_train_data('final_data', is_dots_expanded=False)
#
unet = model.get_unet_model(img_h=192, img_w=256, img_ch=1)
model.train_model(
    model=unet,
    train_data=train_data,
    batch_size=8,
    n_epochs=500,
    model_checkpoint_filename='model_unet_checkpoint',
    patience=100
)