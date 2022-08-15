from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, SpatialDropout2D, Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import MaxPooling2D , Average
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers
from skimage.transform import resize
from tqdm.keras import TqdmCallback
from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras.optimizers import SGD, Adam


from datetime import datetime

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tf.keras.backend.set_image_data_format('channels_last')  # TF dimension ordering in this code

VALIDATION_SIZE_SPLIT = 0.2


def gaussian_kernel(size: int, mean: float, std: float):
    """Makes 2D gaussian Kernel for convolution."""

    d = tfp.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))

    gauss_kernel = tf.einsum(
        'i,j->ij',
        vals,
        vals
    )

    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def dice_coef(target, prediction, axis=(0, 1), smooth=0.0001):
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


def dice_coef_loss(target, prediction, axis=(0, 1), smooth=0.0001):
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


def soft_dice_coef(target, prediction, axis=(0, 1), smooth=0.0001):
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


def combined_dice_ce_loss(target, prediction, weight_dice_loss=0.85, axis=(0, 1), smooth=0.0001):
    """
    Combined Dice and Binary Cross Entropy Loss
    """
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return weight_dice_loss * dice_coef_loss(target, prediction, axis, smooth) + \
           (1 - weight_dice_loss) * bce(target, prediction)

def mae_metric(y_true, y_pred, axis=(0, 1)):
    return tf.abs(
        tf.reduce_sum(y_true, axis=axis) - tf.reduce_sum(y_pred, axis=axis)
    )

def get_csrnet_model():
    sgd = SGD(lr=1e-7, decay=5 * 1e-4, momentum=0.95)
    optimizer = Adam(lr=1e-5)
    vgg16_model = VGG16(weights='imagenet', include_top=False)
    # model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
    x = vgg16_model.get_layer('block4_conv3').output
    x = BatchNormalization()(x)
    #     x = UpSampling2D(size=(8, 8))(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=2, padding='same', use_bias=False,
               kernel_initializer=RandomNormal(stddev=0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=2, padding='same', use_bias=False,
               kernel_initializer=RandomNormal(stddev=0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), dilation_rate=2, padding='same', use_bias=False,
               kernel_initializer=RandomNormal(stddev=0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), dilation_rate=2, padding='same', use_bias=False,
               kernel_initializer=RandomNormal(stddev=0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), dilation_rate=2, padding='same', use_bias=False,
               kernel_initializer=RandomNormal(stddev=0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), dilation_rate=2, padding='same', use_bias=False,
               kernel_initializer=RandomNormal(stddev=0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=1, kernel_size=(1, 1), dilation_rate=1, padding='same', use_bias=True,
               kernel_initializer=RandomNormal(stddev=0.01))(x)
    #     x = BatchNormalization()(x)
    x = Activation('relu')(x)
    model = Model(inputs=vgg16_model.input, outputs=x)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[mae_metric]
    )

    return model

def get_unet_model(img_h=96, img_w=128, img_ch=1):
    inputs = Input((img_h, img_w, img_ch), name='model_image_input')
    s = inputs

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
    c9 = BatchNormalization()(c9)
    c9 = SpatialDropout2D(0.4)(c9)

    masks = Conv2D(1, (1, 1), activation='sigmoid', name='mask_output')(c9)
    # Count training phase
    masks_ = masks

    mask_lv1 = Conv2D(
        32, (3, 3),
        activation='relu',
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(0.0001)
    )(masks_)
    mask_lv1 = SpatialDropout2D(0.4)(mask_lv1)

    mask_lv2 = Conv2D(
        32, (3, 3),
        activation='relu',
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=regularizers.l2(0.0001)
    )(mask_lv1)
    mask_lv2 = SpatialDropout2D(0.4)(mask_lv2)

    mask_average_layer = Average()([masks_, mask_lv1, mask_lv2])
    mask_average_layer = MaxPooling2D(pool_size=(2, 2))(mask_average_layer)

    count_flatten1 = Flatten()(mask_average_layer)

    count_fc1 = Dense(
        32,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.0001)
    )(count_flatten1)
    count_fc1 = Dropout(0.5)(count_fc1)

    count_fc1 = Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.0001)
    )(count_fc1)
    count_fc1 = Dropout(0.5)(count_fc1)

    counts = Dense(1, name='count_output')(count_fc1)

    model = Model(inputs=[inputs], outputs=[counts, masks], name="UNet_V1_Vehicle_Counting")

    loss_weight = 0.6
    model.compile(
        optimizer='adam',
        loss={
            'count_output': tf.keras.losses.MeanAbsoluteError(),
            'mask_output': dice_coef_loss
        },
        loss_weights={
            'count_output': loss_weight,
            'mask_output': 1.0 - loss_weight
        },
        metrics={
            'count_output': [tf.keras.metrics.MeanAbsoluteError()],
            'mask_output': [dice_coef]
        }
    )

    return model


def get_early_stopping(patience=10, verbose=True):
    return EarlyStopping(monitor='val_loss', patience=patience, verbose=verbose, restore_best_weights=True)


def get_model_checkpoint(verbose=True, model_checkpoint_filename='model_unet_checkpoint'):
    now = datetime.now()
    string_date_time = now.strftime('%m_%d_%Y_%H%M%S')
    model_save_file_name = './model_checkpoints/' + model_checkpoint_filename + f'_{string_date_time}' + '.h5'
    return ModelCheckpoint(
        model_save_file_name,
        verbose=verbose,
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )


def get_model_logging(model_log_dir='./logs'):
    return TensorBoard(log_dir=model_log_dir, write_graph=False, write_images=True)


def train_model(model, train_data, valid_data=None, batch_size=64, n_epochs=100,
                model_checkpoint_filename='model_unet_checkpoint', patience=10):
    model_checkpoint = get_model_checkpoint(model_checkpoint_filename=model_checkpoint_filename)
    early_stopping = get_early_stopping(patience=patience)
    tensorboards = get_model_logging()

    model.summary()

    X_train = train_data['train_data']
    Y_train = train_data['train_label_data']
    Y_train_count = train_data['train_count_label_data']

    validation_data = None
    if (valid_data != None):
        validation_data = (valid_data['val_data'], valid_data['val_label_data'])

    if (validation_data != None):
        results = model.fit(
            x=X_train,
            y={
                'count_output': Y_train_count,
                'mask_output': Y_train
            },
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=n_epochs,
            verbose=0,
            callbacks=[early_stopping, model_checkpoint, tensorboards, TqdmCallback(verbose=2)]
        )
    else:
        results = model.fit(
            x=X_train,
            y={
                'count_output': Y_train_count,
                'mask_output': Y_train
            },
            validation_split=VALIDATION_SIZE_SPLIT,
            batch_size=batch_size,
            epochs=n_epochs,
            verbose=0,
            callbacks=[early_stopping, model_checkpoint, tensorboards, TqdmCallback(verbose=2)]
        )

    return results


def load_pretrained_model(model_filename):
    """
    Load a model from Keras file
    """

    custom_objects = {
        "combined_dice_ce_loss": combined_dice_ce_loss,
        "dice_coef_loss": dice_coef_loss,
        "dice_coef": dice_coef,
        "soft_dice_coef": soft_dice_coef,
        "mean_absolute_error": tf.keras.losses.MeanAbsoluteError
    }

    model = tf.keras.models.load_model(model_filename, custom_objects=custom_objects)

    return model


def evaluate_model(model_filename, test_data):
    """
    Evaluate the best model on the validation dataset
    """

    X_test = test_data['test_data']
    Y_test = test_data['test_label_data']
    Y_test_count = test_data['test_count_label_data']

    custom_objects = {
        "combined_dice_ce_loss": combined_dice_ce_loss,
        "dice_coef_loss": dice_coef_loss,
        "dice_coef": dice_coef,
        "soft_dice_coef": soft_dice_coef,
        "mean_absolute_error": tf.keras.losses.MeanAbsoluteError
    }

    model = tf.keras.models.load_model(model_filename, custom_objects=custom_objects)

    loss_weight = 0.8
    model.compile(
        optimizer='adam',
        loss={
            'count_output': tf.keras.losses.MeanAbsoluteError(),
            'mask_output': dice_coef_loss
        },
        metrics={
            'count_output': [tf.keras.metrics.MeanAbsoluteError()],
            'mask_output': [dice_coef, soft_dice_coef]
        }
    )

    print("Evaluating model on test data. Please wait...")
    metrics = model.evaluate(
        x=X_test,
        y={
            'count_output': Y_test_count,
            'mask_output': Y_test
        },
        batch_size=8,
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
    img = (img - img.mean()) / img.std()
    img = np.expand_dims(img, axis=0)

    count_output, mask_output = model.predict(img)
    density_amount = count_output[0][0]
    mask = np.around(mask_output[0])

    return density_amount, mask