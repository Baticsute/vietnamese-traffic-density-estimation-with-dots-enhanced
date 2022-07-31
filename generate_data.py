from utils import data_loader

data_loader.get_train_data('final_data', img_h=96, img_w=128, img_ch=1)
data_loader.get_test_data('final_data', img_h=96, img_w=128, img_ch=1)