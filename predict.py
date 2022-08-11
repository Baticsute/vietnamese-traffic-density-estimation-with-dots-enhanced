import matplotlib.pyplot as plt
from skimage.io import imread
from utils import model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

image = imread('./examples/Image3759.jpg')[:, :, :1]

model.predict('./model_checkpoints/model_unet_checkpoint_08_10_2022_024605.h5', image)