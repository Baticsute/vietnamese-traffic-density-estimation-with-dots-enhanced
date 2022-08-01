import matplotlib.pyplot as plt

from utils import data_loader

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


train_data = data_loader.load_train_data('final_data', is_dots_expanded=True, expand_size=2)
test_data = data_loader.load_test_data('final_data', is_dots_expanded=True, expand_size=2)

s1 = train_data['train_data'][0]
l1 = train_data['train_label_data'][0]

plt.figure(figsize=(30, 15))
plt.imshow(s1)
plt.show()

plt.figure(figsize=(30, 15))
plt.imshow(l1)
plt.show()

# unet = unet_model.get_unet_model(img_h=96, img_w=128, img_ch=1)
# unet_model.train_model(model=unet, train_data=train_data, batch_size=8, n_epochs=10)

# Check if training data looks all right
# ix = random.randint(0, len(train_ids))
# imshow(X_train[ix])
# plt.show()
# imshow(np.squeeze(Y_train[ix]))
# plt.show()