import matplotlib.pyplot as plt

from utils import data_loader
from utils import model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# train_data = data_loader.load_train_data('final_data', is_dots_expanded=False)
test_data = data_loader.load_test_data('final_data', is_dots_expanded=False)

model.evaluate_model('./model_checkpoints/model_unet_checkpoint_08_14_2022_022732.h5', test_data)