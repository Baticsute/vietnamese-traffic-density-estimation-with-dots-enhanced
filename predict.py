import matplotlib.pyplot as plt
from skimage.io import imread
from utils import model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import numpy as np
from tensorflow.keras import Model

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


image = imread('./examples/Image3759.jpg')[:, :, :1]
X_test = np.load('./data_storage/final_data/X_test.npy')

model_pretrained = model.load_pretrained_model('./model_checkpoints/model_unet_checkpoint_08_14_2022_022732.h5')
model_pretrained.summary()
results = []
model_pretrained = Model(inputs=model_pretrained.inputs, outputs=model_pretrained.layers[58].output)

for i in range(2):
    img = np.expand_dims(X_test[i], axis=0)
    results.append(model_pretrained.predict(img))

print(results)