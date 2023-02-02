import math
import gc

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization, SpatialDropout2D, \
    Dense, Dropout, \
    Flatten, Activation, Multiply, concatenate, MaxPooling2D, Average, Lambda, GlobalAveragePooling2D, Add, Subtract, multiply, add, ReLU

from utils.pooling import MaxUnpooling2D, MaxPoolingWithArgmax2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers
from skimage.transform import resize
from tqdm.keras import TqdmCallback
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

from tensorflow.keras.metrics import MeanAbsoluteError as mae_metric, MeanSquaredError as mse_metric
from tensorflow.keras.losses import MeanAbsoluteError as mae_error, MeanSquaredError as mse_error

from datetime import datetime

import tensorflow as tf
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
tf.keras.backend.set_image_data_format('channels_last')  # TF dimension ordering in this code

VALIDATION_SIZE_SPLIT = 0.2


def dice_coef(target, prediction, axis=(1, 2), smooth=0.0001):
    """
    Sorenson Dice
    \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
    where T is ground truth mask and P is the prediction mask
    """
    prediction = tf.round(prediction)  # Round to 0 or 1

    intersection = tf.reduce_sum(target * prediction, axis=axis)
    union = tf.reduce_sum(target + prediction, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator

    return tf.reduce_mean(coef)


def dice_coef_loss(target, prediction, axis=(1, 2), smooth=0.0001):
    """
    Sorenson (Soft) Dice loss
    Using -log(Dice) as the loss since it is better behaved.
    Also, the log allows avoidance of the division which
    can help prevent underflow when the numbers are very small.
    """
    intersection = tf.reduce_sum(prediction * target, axis=axis)
    p = tf.reduce_sum(prediction, axis=axis)
    t = tf.reduce_sum(target, axis=axis)
    numerator = tf.reduce_mean(intersection + smooth)
    denominator = tf.reduce_mean(t + p + smooth)
    dice_loss = -tf.math.log(2. * numerator) + tf.math.log(denominator)

    return dice_loss


def soft_dice_coef(target, prediction, axis=(1, 2), smooth=0.0001):
    """
    Sorenson (Soft) Dice  - Don't round the predictions
    \frac{  2 \times \left | T \right | \cap \left | P \right |}{ \left | T \right | +  \left | P \right |  }
    where T is ground truth mask and P is the prediction mask
    """

    intersection = tf.reduce_sum(target * prediction, axis=axis)
    union = tf.reduce_sum(target + prediction, axis=axis)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    coef = numerator / denominator

    return tf.reduce_mean(coef)


def combined_dice_ce_loss(target, prediction, weight_dice_loss=0.85, axis=(1, 2), smooth=0.0001):
    """
    Combined Dice and Binary Cross Entropy Loss
    """
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return weight_dice_loss * dice_coef_loss(target, prediction, axis, smooth) + \
           (1 - weight_dice_loss) * bce(target, prediction)


def density_mae(y_true, y_pred, axis=(1, 2, 3)):
    return tf.abs(
        tf.reduce_sum(y_true, axis=axis) - tf.reduce_sum(y_pred, axis=axis)
    )

def density_mse(y_true, y_pred, axis=(1, 2, 3)):
    return tf.square(
        tf.reduce_sum(y_true, axis=axis) - tf.reduce_sum(y_pred, axis=axis)
    )

def count_mae(y_true, y_pred):
    return tf.abs(
        tf.reduce_sum(y_true) - tf.reduce_sum(y_pred)
    )

def count_mse(y_true, y_pred):
    return tf.square(
        tf.reduce_sum(y_true) - tf.reduce_sum(y_pred)
    )


def loss_euclidean_distance(y_true, y_pred):
    """ Computes the euclidean distance between two tensors.
    The euclidean distance or $L^2$ distance between points $p$ and $q$ is the length of the line segment
    connecting them.
    $$
    distance(q,p) =\\sqrt{\\sum_{i=1}^{n}\\left(q_{i}-p_{i}\\right)^{2}}
    $$
    Args:
        y_true: a ``Tensor``
        y_pred: a ``Tensor``
        dim: dimension along which the euclidean distance is computed
    Returns:
        ``Tensor``: a ``Tensor`` with the euclidean distances between the two tensors
    """
    # distance = tf.sqrt(
    #     tf.reduce_sum(
    #         tf.square(
    #             tf.subtract(y_pred, y_true)
    #         ),
    #         axis=-1
    #     )
    # )
    # return distance

    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def MAE_BCE(y_true, y_pred, alpha=1, beta=1):
    mae = K.mean(K.abs(y_true - y_pred), axis=-1)
    bce = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return alpha * mae + beta * bce

def MSE_BCE(alpha=1000, beta=10):
    def mse_bce(y_true, y_pred):
        mse = K.mean(K.square(y_true - y_pred), axis=-1)
        bce = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        return alpha * mse + beta * bce
    return mse_bce

def EUCLID_BCE(y_true, y_pred, alpha=100, beta=10):
    euclid = K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)))
    bce = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return alpha * euclid + beta * bce

def focal_loss(targets, inputs, alpha=0.8, gamma=2):
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    bce = K.binary_crossentropy(targets, inputs)
    bce_exp = K.exp(-bce)
    loss = K.mean(alpha * K.pow((1 - bce_exp), gamma) * bce)

    return loss

def get_wnet_model(img_h=96, img_w=128, img_ch=1, BN=False, is_multi_output=False):
    # Difference with original paper: padding 'valid vs same'
    conv_kernel_initializer = RandomNormal(stddev=0.01)
    adam_optimizer = Adam(lr=1e-4, decay=5e-3)
    rms = RMSprop(lr=1e-4, momentum=0.7, decay=0.0001)

    input_flow = Input((img_h, img_w, img_ch), name='model_image_input')
    dtype = tf.float32
    x = Lambda(
        lambda batch: (batch - tf.constant([0.485, 0.456, 0.406], dtype=dtype)) / tf.constant([0.229, 0.224, 0.225],
                                                                                              dtype=dtype))(input_flow)
    # Encoder
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)

    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x_1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x_1 = BatchNormalization()(x_1) if BN else x_1
    x_1 = Activation('relu')(x_1)

    x = MaxPooling2D((2, 2))(x_1)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x_2 = BatchNormalization()(x_2) if BN else x_2
    x_2 = Activation('relu')(x_2)

    x = MaxPooling2D((2, 2))(x_2)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x_3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x_3 = BatchNormalization()(x_3) if BN else x_3
    x_3 = Activation('relu')(x_3)

    x = MaxPooling2D((2, 2))(x_3)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x_4 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x_4 = BatchNormalization()(x_4) if BN else x_4
    x_4 = Activation('relu')(x_4)

    # Decoder 1
    x = UpSampling2D((2, 2))(x_4)
    x = concatenate([x_3, x])
    x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)

    x = UpSampling2D((2, 2))(x)
    x = concatenate([x_2, x])
    x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)

    x = UpSampling2D((2, 2))(x)
    x = concatenate([x_1, x])
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)

    # Decoder 2
    x_rb = UpSampling2D((2, 2))(x_4)
    x_rb = concatenate([x_3, x_rb])
    x_rb = Conv2D(256, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x_rb)
    x_rb = BatchNormalization()(x_rb) if BN else x_rb
    x_rb = Activation('relu')(x_rb)
    x_rb = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x_rb)
    x_rb = BatchNormalization()(x_rb) if BN else x_rb
    x_rb = Activation('relu')(x_rb)

    x_rb = UpSampling2D((2, 2))(x_rb)
    x_rb = concatenate([x_2, x_rb])
    x_rb = Conv2D(128, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x_rb)
    x_rb = BatchNormalization()(x_rb) if BN else x_rb
    x_rb = Activation('relu')(x_rb)
    x_rb = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x_rb)
    x_rb = BatchNormalization()(x_rb) if BN else x_rb
    x_rb = Activation('relu')(x_rb)

    x_rb = UpSampling2D((2, 2))(x_rb)
    x_rb = concatenate([x_1, x_rb])
    x_rb = Conv2D(64, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x_rb)
    x_rb = BatchNormalization()(x_rb) if BN else x_rb
    x_rb = Activation('relu')(x_rb)
    x_rb = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x_rb)
    x_rb = BatchNormalization()(x_rb) if BN else x_rb
    x_rb = Activation('relu')(x_rb)
    x_rb = Conv2D(32, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x_rb)
    x_rb = BatchNormalization()(x_rb) if BN else x_rb
    x_rb = Activation('relu')(x_rb)
    x_rb = Conv2D(1, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer,
                  activation='sigmoid')(x_rb)  # Sigmoid activation

    # Multiplication
    x = multiply([x, x_rb])
    x = Conv2D(1, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer,
               activation='relu')(x)

    model = Model(inputs=input_flow, outputs=x)

    front_end = VGG16(weights='imagenet', include_top=False)
    weights_front_end = []
    for layer in front_end.layers:
        if 'conv' in layer.name:
            weights_front_end.append(layer.get_weights())
    counter_conv = 0
    for i in range(len(model.layers)):
        if counter_conv >= 13:
            break
        if 'conv' in model.layers[i].name:
            model.layers[i].set_weights(weights_front_end[counter_conv])
            counter_conv += 1

    model.compile(
        optimizer=adam_optimizer,
        loss=MSE_BCE,
        metrics=[density_mae, density_mse]
    )

    return model


def get_csrnet_model(img_h=480, img_w=640, img_ch=1, is_multi_output=False):

    input_flow = Input((img_h, img_w, img_ch), name='model_image_input')
    dilated_conv_kernel_initializer = RandomNormal(stddev=0.01)

    # front-end
    dtype = tf.float32
    x = Lambda(
        lambda batch: (batch - tf.constant([0.485, 0.456, 0.406], dtype=dtype)) / tf.constant([0.229, 0.224, 0.225],
                                                                                              dtype=dtype))(input_flow)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    # back-end
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu',
               kernel_initializer=dilated_conv_kernel_initializer, name='2dilation_conv2D_1')(x)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu',
               kernel_initializer=dilated_conv_kernel_initializer, name='2dilation_conv2D_2')(x)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu',
               kernel_initializer=dilated_conv_kernel_initializer, name='2dilation_conv2D_3')(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu',
               kernel_initializer=dilated_conv_kernel_initializer, name='2dilation_conv2D_4')(x)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu',
               kernel_initializer=dilated_conv_kernel_initializer, name='2dilation_conv2D_5')(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu',
               kernel_initializer=dilated_conv_kernel_initializer, name='2dilation_conv2D_6')(x)

    density_map = Conv2D(1, 1, strides=(1, 1), padding='same', activation='relu',
                         kernel_initializer=dilated_conv_kernel_initializer, name='density_map')(x)

    # count_branch = Conv2D(1, 1, strides=(1, 1), padding='same', activation='relu',
    #                      kernel_initializer=dilated_conv_kernel_initializer, name='count_branch')(x)

    flatten = Flatten(name='flatten')(density_map)
    dense128 = Dense(128, activation=None)(flatten)
    dense64 = Dense(64, activation=None)(dense128)
    dense32 = Dense(32, activation=None)(dense64)
    count_output_flow = Dense(1, activation='linear', name="estimation")(dense32)

    if is_multi_output:
        model = Model(inputs=input_flow, outputs={"density_map": density_map, "estimation": count_output_flow})
    else:
        model = Model(inputs=input_flow, outputs=density_map)

    front_end = VGG16(weights='imagenet', include_top=False)

    weights_front_end = []
    for layer in front_end.layers:
        if 'conv' in layer.name:
            weights_front_end.append(layer.get_weights())
    counter_conv = 0
    for i in range(len(front_end.layers)):
        if counter_conv >= 10:
            break
        if 'conv' in model.layers[i].name:
            model.layers[i].set_weights(weights_front_end[counter_conv])
            counter_conv += 1

    sgd = SGD(lr=1e-7, decay=(5 * 1e-4), momentum=0.95)
    rms = RMSprop(lr=1e-4, momentum=0.7, decay=0.0001)
    adam_optimizer = Adam(lr=1e-6)

    losses = None
    lossWeights = None
    metrics = None
    if is_multi_output:
        losses = {
            'density_map': 'binary_crossentropy',
            'estimation': mae_error()
        }
        lossWeights = {"density_map": 0.7, "estimation": 0.3}

        metrics = {
            'density_map': [density_mae, density_mse],
            'estimation': [mae_metric(), mse_metric()]
        }

        model.compile(
            optimizer=sgd,
            loss=losses,
            loss_weights=lossWeights,
            metrics=metrics
        )
    else:
        model.compile(
            optimizer=adam_optimizer,
            loss=MSE_BCE,
            metrics=[density_mae, density_mse]
        )

    return model

def w_by_sigmoid_normalization(x):
    return tf.math.atan(K.sigmoid(x)) * (2.0 / math.pi)

def get_u_asd_net(img_h=480, img_w=640, img_ch=1, is_multi_output=False, BN=True, is_use_max_unpool=True):

    input_flow = Input((img_h, img_w, img_ch), name='model_image_input')
    conv_kernel_initializer = RandomNormal(stddev=0.01)

    # front-end (backbone) aka encoder
    dtype = tf.float32
    x = Lambda(
        lambda batch: (batch - tf.constant([0.485, 0.456, 0.406], dtype=dtype)) / tf.constant([0.229, 0.224, 0.225],
                                                                                              dtype=dtype))(input_flow)

    c1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(x)
    c1 = ReLU()(c1)
    c1 = BatchNormalization()(c1) if BN else c1
    c1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(c1)
    c1 = ReLU()(c1)
    c1 = BatchNormalization()(c1) if BN else c1
    p1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c1)

    c2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(p1)
    c2 = ReLU()(c2)
    c2 = BatchNormalization()(c2) if BN else c2
    fm1_idx1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(c2)
    fm1_idx1 = ReLU()(fm1_idx1)
    fm1_idx1 = BatchNormalization()(fm1_idx1) if BN else fm1_idx1
    p2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(fm1_idx1)

    c3 = Conv2D(256, (3, 3), strides=(1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(p2)
    c3 = ReLU()(c3)
    c3 = BatchNormalization()(c3) if BN else c3
    c3 = Conv2D(256, (3, 3), strides=(1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(c3)
    c3 = ReLU()(c3)
    c3 = BatchNormalization()(c3) if BN else c3
    fm2_idx2 = Conv2D(256, (3, 3), strides=(1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(c3)
    fm2_idx2 = ReLU()(fm2_idx2)
    fm2_idx2 = BatchNormalization()(fm2_idx2) if BN else fm2_idx2
    p3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(fm2_idx2)

    c4 = Conv2D(512, (3, 3), strides=(1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(p3)
    c4 = ReLU()(c4)
    c4 = BatchNormalization()(c4) if BN else c4
    c4 = Conv2D(512, (3, 3), strides=(1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(c4)
    c4 = ReLU()(c4)
    c4 = BatchNormalization()(c4) if BN else c4
    fm3_idx3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(c4)
    fm3_idx3 = ReLU()(fm3_idx3)
    fm3_idx3 = BatchNormalization()(fm3_idx3) if BN else fm3_idx3
    p4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(fm3_idx3)

    c5 = Conv2D(512, (3, 3), strides=(1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(p4)
    c5 = ReLU()(c5)
    c5 = BatchNormalization()(c5) if BN else c5
    c5 = Conv2D(512, (3, 3), strides=(1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(c5)
    c5 = ReLU()(c5)
    c5 = BatchNormalization()(c5) if BN else c5
    fm4_idx4 = Conv2D(512, (3, 3), strides=(1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(c5)
    fm4_idx4 = ReLU()(fm4_idx4)
    fm4_idx4 = BatchNormalization()(fm4_idx4) if BN else fm4_idx4
    if is_use_max_unpool:
        fm4_idx4_pm, fm4_idx4_pm_argmax = MaxPoolingWithArgmax2D(pool_size=(2, 2), strides=(2, 2))(fm4_idx4)
        upc5 = MaxUnpooling2D(up_size=(4, 4))([fm4_idx4_pm, fm4_idx4_pm_argmax])
    else:
        upc5 = UpSampling2D(size=(2, 2))(fm4_idx4)

    c6 = concatenate([fm3_idx3, upc5])
    c6 = Conv2D(256, (1, 1), strides=(1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(c6)
    c6 = ReLU()(c6)
    c6 = BatchNormalization()(c6) if BN else c6
    c6 = Conv2D(256, (3, 3), strides=(1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(c6)
    c6 = ReLU()(c6)
    c6 = BatchNormalization()(c6) if BN else c6
    if is_use_max_unpool:
        c6_mp, c6_mp_argmax = MaxPoolingWithArgmax2D(pool_size=(2, 2), strides=(2, 2))(c6)
        upc6 = MaxUnpooling2D(up_size=(4, 4))([c6_mp, c6_mp_argmax])
    else:
        upc6 = UpSampling2D(size=(2, 2))(c6)

    c7 = concatenate([fm2_idx2, upc6])
    c7 = Conv2D(128, (1, 1), strides=(1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(c7)
    c7 = ReLU()(c7)
    c7 = BatchNormalization()(c7) if BN else c7
    c7 = Conv2D(128, (3, 3), strides=(1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(c7)
    c7 = ReLU()(c7)
    c7 = BatchNormalization()(c7) if BN else c7
    if is_use_max_unpool:
        c7_mp, c7_mp_argmax = MaxPoolingWithArgmax2D(pool_size=(2, 2), strides=(2, 2))(c7)
        upc7 = MaxUnpooling2D(up_size=(4, 4))([c7_mp, c7_mp_argmax])
    else:
        upc7 = UpSampling2D(size=(2, 2))(c7)

    c8 = concatenate([fm1_idx1, upc7])
    c8 = Conv2D(64, (1, 1), strides=(1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(c8)
    c8 = ReLU()(c8)
    c8 = BatchNormalization()(c8) if BN else c8
    c8 = Conv2D(64, (3, 3), strides=(1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(c8)
    c8 = ReLU()(c8)
    c8 = BatchNormalization()(c8) if BN else c8
    c8 = Conv2D(32, (3, 3), strides=(1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(c8)
    c8 = ReLU()(c8)

    #ASD Module
    asd_b1 = Conv2DTranspose(512, (1, 1), strides=(2, 2), padding='same')(fm4_idx4)
    # asd_b1 = Conv2D(512, (1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(asd_b1)
    asd_b1 = Conv2D(256, (1, 1), padding='same',  kernel_initializer=conv_kernel_initializer)(asd_b1)
    asd_b1 = ReLU()(asd_b1)
    asd_b1 = Conv2D(128, (7, 7), padding='same',  kernel_initializer=conv_kernel_initializer)(asd_b1)
    asd_b1 = ReLU()(asd_b1)
    asd_b1 = Conv2D(64, (7, 7), padding='same',  kernel_initializer=conv_kernel_initializer)(asd_b1)
    asd_b1 = ReLU()(asd_b1)
    asd_b1 = Conv2D(1, (3, 3), padding='same',  kernel_initializer=conv_kernel_initializer)(asd_b1)
    asd_b1 = ReLU()(asd_b1)
    asd_b1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(asd_b1)

    # asd_b2 = Conv2D(256, (3, 3), padding='same',  kernel_initializer=conv_kernel_initializer)(fm4_idx4)
    # asd_b2 = ReLU()(asd_b2)
    asd_b2 = Conv2D(128, (3, 3), padding='same',  kernel_initializer=conv_kernel_initializer)(fm4_idx4)
    asd_b2 = ReLU()(asd_b2)
    asd_b2 = Conv2D(64, (3, 3), padding='same',  kernel_initializer=conv_kernel_initializer)(asd_b2)
    asd_b2 = ReLU()(asd_b2)
    asd_b2 = Conv2D(1, (3, 3), padding='same',  kernel_initializer=conv_kernel_initializer)(asd_b2)
    asd_b2 = ReLU()(asd_b2)

    asd_b3 = GlobalAveragePooling2D()(fm4_idx4)
    asd_b3 = Flatten()(asd_b3)
    asd_b3 = Dense(32, activation=None)(asd_b3)
    w = Dense(1)(asd_b3)
    # w = Activation(w_by_sigmoid_normalization)(w)
    w = Activation('sigmoid')(w)
    w = Lambda(lambda _w: tf.math.atan(_w) * (2.0 / math.pi))(w)

    one_subtract_w = Lambda(lambda _w: 1.0 - _w)(w)
    asd_b1_w = multiply([one_subtract_w, asd_b1])
    asd_b2_w = multiply([w, asd_b2])
    asd_b1_b2 = add([asd_b1_w, asd_b2_w])
    asd_final = UpSampling2D((8, 8), interpolation='nearest')(asd_b1_b2)

    density_map = multiply([c8, asd_final])
    density_map = Conv2D(1, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(density_map)
    density_map = ReLU()(density_map)

    model = Model(inputs=input_flow, outputs=density_map)

    front_end = VGG16(weights='imagenet', include_top=False)

    weights_front_end = []
    for layer in front_end.layers:
        if 'conv' in layer.name:
            weights_front_end.append(layer.get_weights())
    counter_conv = 0
    for i in range(len(front_end.layers)):
        if counter_conv >= 13:
            break
        if 'conv' in model.layers[i].name:
            model.layers[i].set_weights(weights_front_end[counter_conv])
            counter_conv += 1

    sgd = SGD(lr=1e-7, decay=(5 * 1e-4), momentum=0.95)
    rms = RMSprop(lr=1e-4, momentum=0.7, decay=0.0001)
    adam_optimizer = Adam(lr=1e-6)

    model.compile(
        optimizer=adam_optimizer,
        loss=MSE_BCE(alpha=1000, beta=20),
        metrics=[density_mae, density_mse],
        run_eagerly=True
    )

    return model

def get_unet_model(img_h=96, img_w=128, img_ch=1):
    inputs = Input((img_h, img_w, img_ch), name='model_image_input')
    s = inputs
    adam_optimizer = Adam(lr=1e-6)
    # sgd_optimizer = SGD(lr=1e-7, decay=(5 * 1e-4), momentum=0.95)

    init = RandomNormal(stddev=0.01)

    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = SpatialDropout2D(0.4)(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = SpatialDropout2D(0.4)(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = SpatialDropout2D(0.4)(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = SpatialDropout2D(0.4)(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = SpatialDropout2D(0.4)(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = BatchNormalization()(c6)
    c6 = SpatialDropout2D(0.4)(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c7 = BatchNormalization()(c7)
    c7 = SpatialDropout2D(0.4)(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    c8 = BatchNormalization()(c8)
    c8 = SpatialDropout2D(0.4)(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    # c9 = BatchNormalization()(c9)
    # c9 = SpatialDropout2D(0.4)(c9)

    # back-end
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu',
               kernel_initializer=init, name='2dilation_conv2D_1')(c9)

    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu',
               kernel_initializer=init, name='2dilation_conv2D_2')(x)

    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', dilation_rate=2, activation='relu',
               kernel_initializer=init, name='2dilation_conv2D_3')(x)

    masks = Conv2D(1, 1, strides=(1, 1), activation='relu', padding='same', name='mask_output')(x)

    model = Model(inputs=[inputs], outputs=[masks], name="UNet_V1_Vehicle_Counting")

    model.compile(
        optimizer=adam_optimizer,
        loss=focal_loss,
        metrics=[density_mae, density_mse]
    )

    return model


def get_early_stopping(patience=10, verbose=True, monitor='val_loss'):
    return EarlyStopping(monitor=monitor, patience=patience, verbose=verbose, restore_best_weights=True)


def get_model_checkpoint(verbose=True, model_checkpoint_filename='model_unet_checkpoint', monitor='val_loss',
                         mode='min'):
    now = datetime.now()
    string_date_time = now.strftime('%m_%d_%Y_%H%M%S')
    model_save_file_name = './model_checkpoints/' + model_checkpoint_filename + f'_{string_date_time}' + '.h5'
    return ModelCheckpoint(
        model_save_file_name,
        verbose=verbose,
        monitor=monitor,
        save_best_only=True,
        mode=mode
    )


def get_model_logging(model_log_dir='./logs'):
    return TensorBoard(log_dir=model_log_dir, write_graph=False)

class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        K.clear_session()

def train_model(model, train_data, valid_data=None,
                n_epochs=100, steps_per_epoch=None, validation_steps=None,
                model_checkpoint_filename='model_unet_checkpoint', patience=10, monitor='val_loss'):

    model_checkpoint = get_model_checkpoint(model_checkpoint_filename=model_checkpoint_filename, monitor=monitor)
    early_stopping = get_early_stopping(patience=patience, monitor=monitor)
    tensorboards = get_model_logging()

    model.summary()

    model.fit(
        train_data,
        validation_data=valid_data,
        epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=0,
        callbacks=[early_stopping, model_checkpoint, tensorboards, TqdmCallback(verbose=2)],
        use_multiprocessing=True,
        workers=16
    )

    now = datetime.now()
    string_date_time = now.strftime('%m_%d_%Y_%H%M%S')
    model_save_file_name = './model_checkpoints/' + model_checkpoint_filename + f'_{string_date_time}_last_epoch' + '.h5'
    model.save(model_save_file_name)

def train_generator_model(
        model,
        train_data_generator,
        validation_data_generator,
        batch_size=64,
        n_epochs=100,
        model_checkpoint_filename='model_unet_checkpoint',
        patience=10
):
    model_checkpoint = get_model_checkpoint(model_checkpoint_filename=model_checkpoint_filename)
    early_stopping = get_early_stopping(patience=patience)
    tensorboards = get_model_logging()

    model.summary()

    model.fit_generator(
        generator=train_data_generator,
        validation_data=validation_data_generator,
        epochs=n_epochs,
        verbose=0,
        use_multiprocessing=True,
        workers=16,
        callbacks=[early_stopping, model_checkpoint, tensorboards, TqdmCallback(verbose=2)]
    )


def load_pretrained_model(model_filename):
    """
    Load a model from Keras file
    """

    custom_objects = {
        "loss_euclidean_distance": loss_euclidean_distance,
        "density_mae": density_mae,
        "density_mse": density_mse,
        "count_mae": count_mae,
        "count_mse": count_mse,
        "MSE_BCE": MSE_BCE,
        "mse_bce": MSE_BCE,
        "w_by_sigmoid_normalization": w_by_sigmoid_normalization,
        "MaxPoolingWithArgmax2D": MaxPoolingWithArgmax2D,
        "MaxUnpooling2D": MaxUnpooling2D,
    }

    model = tf.keras.models.load_model(model_filename, custom_objects=custom_objects)

    return model


def evaluate_model(model_filename, test_data):
    """
    Evaluate the best model on the validation dataset
    """

    custom_objects = {
        "loss_euclidean_distance": loss_euclidean_distance,
        "density_mae": density_mae,
        "density_mse": density_mse,
        "count_mae": count_mae,
        "count_mse": count_mse,
        "MSE_BCE": MSE_BCE,
        "w_by_sigmoid_normalization": w_by_sigmoid_normalization,
        "MaxPoolingWithArgmax2D": MaxPoolingWithArgmax2D,
        "MaxUnpooling2D": MaxUnpooling2D,
    }

    model = tf.keras.models.load_model(model_filename, custom_objects=custom_objects)


    losses = {
        'density_map_output': 'binary_crossentropy',
        'count_output': count_mae
    }

    _metrics = {
        'density_map_output': [density_mae, density_mse],
        'count_output': [count_mae, count_mse]
    }

    model.compile(
        optimizer='adam',
        loss=losses,
        metrics=_metrics
    )

    print("Evaluating model on test data. Please wait...")
    metrics = model.evaluate(
        test_data,
        verbose=1
    )

    for idx, metric in enumerate(metrics):
        print("Test dataset {} = {:.2f}".format(model.metrics_names[idx], metric))


def predict(model, image):
    input_shape = list(model.get_layer('model_image_input').output_shape[0])
    input_shape[0] = 1
    img_h = input_shape[1]
    img_w = input_shape[2]
    img = resize(image, (img_h, img_w), mode='constant', preserve_range=True)
    # img = img / 255.0
    img = np.expand_dims(img, axis=0)

    mask_output = model.predict(img)
    mask = mask_output[0]

    return mask
