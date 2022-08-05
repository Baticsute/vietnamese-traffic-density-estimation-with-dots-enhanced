from utils import data_loader
from utils import unet_model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import tensorflow as tf

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
tf.keras.backend.set_image_data_format('channels_last')  # TF dimension ordering in this code

train_data = data_loader.load_train_data('final_data', is_dots_expanded=False)
#
unet = unet_model.get_unet_model_v2(img_h=192, img_w=256, img_ch=1, n_feature_maps=32)
unet_model.train_model(
    model=unet,
    train_data=train_data,
    batch_size=8,
    n_epochs=500,
    model_checkpoint_path='./model_checkpoints/model_unet_v2_checkpoint.h5',
    patience=50
)