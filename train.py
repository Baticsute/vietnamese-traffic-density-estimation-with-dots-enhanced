from utils import data_loader
from utils import model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import tensorflow as tf
import math

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
tf.keras.backend.set_image_data_format('channels_last')  # TF dimension ordering in this code

BATCH_SIZE = 4
BATCH_SAMPLE_SIZE = 128
DATASET_LOOP = 10
DOWN_SAMPLING = 2
is_multi_outputs = False
BUFFER_SIZE = 512

dataset_dict = data_loader.load_dataset_paths(dataset_name='night_traffic', validation_split_size=0.0, is_exist_validation=True)

train_input_data = dataset_dict['train']['images']
train_output_data = dataset_dict['train']['density_maps']

validation_input_data = dataset_dict['validation']['images']
validation_output_data = dataset_dict['validation']['density_maps']

train_dataset, train_size = data_loader.load_dataset(
    input_paths=train_input_data,
    output_paths=train_output_data,
    output_type='density_maps',
    batch_size=BATCH_SIZE,
    shuffle=True,
    downsampling_size=DOWN_SAMPLING,
    buffer_size=BUFFER_SIZE,
    is_multi_outputs=is_multi_outputs
)

validation_dataset, val_size = data_loader.load_dataset(
    input_paths=validation_input_data,
    output_paths=validation_output_data,
    output_type='density_maps',
    batch_size=BATCH_SIZE,
    shuffle=False,
    downsampling_size=DOWN_SAMPLING,
    is_multi_outputs=is_multi_outputs
)

# net = model.get_csrnet_model(img_h=480, img_w=640, img_ch=3, is_multi_output=is_multi_outputs)
net = model.get_wnet_model(img_h=480, img_w=640, img_ch=3, BN=False)

# net.summary()

model.train_model(
    model=net,
    train_data=train_dataset,
    valid_data=validation_dataset,
    steps_per_epoch=int(math.ceil((1. * train_size) / BATCH_SAMPLE_SIZE)),
    validation_steps=val_size // BATCH_SIZE,
    n_epochs=BATCH_SAMPLE_SIZE * DATASET_LOOP,
    model_checkpoint_filename='model_w_net_1_dot_night_traffic',
    patience=100,
    monitor='val_loss'
)