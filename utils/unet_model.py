from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras import backend as K

import tensorflow as tf

tf.keras.backend.set_image_data_format('channels_last')  # TF dimension ordering in this code

VALIDATION_SIZE_SPLIT = 0.05


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


def get_unet_model_v2(img_h=96, img_w=128, img_ch=1, n_feature_maps=32):
    concat_axis = -1
    if tf.keras.backend.image_data_format() == 'channels_last':
        concat_axis = -1
    else:
        concat_axis = 1

    params = dict(
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        kernel_initializer="he_uniform"
    )

    params_trans = dict(
        kernel_size=(2, 2),
        strides=(2, 2),
        padding="same"
    )

    inputs = Input((img_h, img_w, img_ch))

    encodeA = Conv2D(name="encodeAa", filters=n_feature_maps, **params)(inputs)
    encodeA = Conv2D(name="encodeAb", filters=n_feature_maps, **params)(encodeA)
    poolA = MaxPooling2D(name="poolA", pool_size=(2, 2))(encodeA)

    encodeB = Conv2D(name="encodeBa", filters=n_feature_maps * 2, **params)(poolA)
    encodeB = Conv2D(name="encodeBb", filters=n_feature_maps * 2, **params)(encodeB)
    poolB = MaxPooling2D(name="poolB", pool_size=(2, 2))(encodeB)

    encodeC = Conv2D(name="encodeCa", filters=n_feature_maps * 4, **params)(poolB)
    encodeC = SpatialDropout2D(0.2)(encodeC)
    encodeC = Conv2D(name="encodeCb", filters=n_feature_maps * 4, **params)(encodeC)
    poolC = MaxPooling2D(name="poolC", pool_size=(2, 2))(encodeC)

    encodeD = Conv2D(name="encodeDa", filters=n_feature_maps * 8, **params)(poolC)
    encodeD = SpatialDropout2D(0.2)(encodeD)
    encodeD = Conv2D(name="encodeDb", filters=n_feature_maps * 8, **params)(encodeD)
    poolD = MaxPooling2D(name="poolD", pool_size=(2, 2))(encodeD)

    encodeE = Conv2D(name="encodeEa", filters=n_feature_maps * 16, **params)(poolD)
    encodeE = Conv2D(name="encodeEb", filters=n_feature_maps * 16, **params)(encodeE)
    up = Conv2DTranspose(name="transconvE", filters=n_feature_maps * 8, **params_trans)(encodeE)

    concatD = concatenate([up, encodeD], axis=concat_axis, name="concatD")

    decodeC = Conv2D(name="decodeCa", filters=n_feature_maps * 8, **params)(concatD)
    decodeC = Conv2D(name="decodeCb", filters=n_feature_maps * 8, **params)(decodeC)

    up = Conv2DTranspose(name="transconvC", filters=n_feature_maps * 4, **params_trans)(decodeC)

    concatC = concatenate([up, encodeC], axis=concat_axis, name="concatC")

    decodeB = Conv2D(name="decodeBa", filters=n_feature_maps * 4, **params)(concatC)
    decodeB = Conv2D(name="decodeBb", filters=n_feature_maps * 4, **params)(decodeB)

    up = Conv2DTranspose(name="transconvB", filters=n_feature_maps * 2, **params_trans)(decodeB)
    concatB = concatenate([up, encodeB], axis=concat_axis, name="concatB")

    decodeA = Conv2D(name="decodeAa", filters=n_feature_maps * 2, **params)(concatB)
    decodeA = Conv2D(name="decodeAb", filters=n_feature_maps * 2, **params)(decodeA)

    up = Conv2DTranspose(name="transconvA", filters=n_feature_maps, **params_trans)(decodeA)

    concatA = concatenate([up, encodeA], axis=concat_axis, name="concatA")

    convOut = Conv2D(name="convOuta", filters=n_feature_maps, **params)(concatA)
    convOut = Conv2D(name="convOutb", filters=n_feature_maps, **params)(convOut)

    prediction = Conv2D(
        name="PredictionMask",
        filters=1, kernel_size=(1, 1),
        activation="sigmoid"
    )(convOut)

    model = Model(inputs=[inputs], outputs=[prediction], name="UNet_Vehicle_Counting")
    model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef, soft_dice_coef])

    return model


def get_unet_model(img_h=96, img_w=128, img_ch=1):
    inputs = Input((img_h, img_w, img_ch))
    s = inputs

    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = BatchNormalization()(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = BatchNormalization()(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c7 = BatchNormalization()(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    c8 = BatchNormalization()(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    c9 = BatchNormalization()(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef, soft_dice_coef])

    return model


def get_early_stopping(patience=10, verbose=True):
    return EarlyStopping(patience=patience, verbose=verbose, restore_best_weights=True)


def get_model_checkpoint(verbose=True, model_checkpoint_path='./model_checkpoints/model_unet_checkpoint.h5'):
    return ModelCheckpoint(model_checkpoint_path, verbose=verbose, monitor='val_loss', save_best_only=True)


def get_model_logging(model_log_dir='./logs'):
    return TensorBoard(log_dir=model_log_dir, write_graph=True, write_images=True)


def train_model(model, train_data, valid_data=None, batch_size=64, n_epochs=100):
    model_checkpoint = get_model_checkpoint()
    early_stopping = get_early_stopping()
    tensorboards = get_model_logging()

    model.summary()

    X_train = train_data['train_data']
    Y_train = train_data['train_label_data']

    validation_data = None
    if (valid_data != None):
        validation_data = (valid_data['val_data'], valid_data['val_label_data'])

    if (validation_data != None):
        results = model.fit(
            X_train,
            Y_train,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=n_epochs,
            callbacks=[early_stopping, model_checkpoint, tensorboards]
        )
    else:
        results = model.fit(
            X_train,
            Y_train,
            validation_split=VALIDATION_SIZE_SPLIT,
            batch_size=batch_size,
            epochs=n_epochs,
            callbacks=[early_stopping, model_checkpoint, tensorboards]
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
        "soft_dice_coef": soft_dice_coef
    }
    model = tf.keras.models.load_model(model_filename, custom_objects=custom_objects)

    model.compile(
        optimizer='adam',
        loss=dice_coef_loss,
        metrics=['acc', combined_dice_ce_loss, dice_coef_loss, dice_coef, soft_dice_coef]
    )

    return model


def evaluate_model(model_filename, test_data):
    """
    Evaluate the best model on the validation dataset
    """

    X_test = test_data['test_data']
    Y_test = test_data['test_label_data']

    custom_objects = {
        "combined_dice_ce_loss": combined_dice_ce_loss,
        "dice_coef_loss": dice_coef_loss,
        "dice_coef": dice_coef,
        "soft_dice_coef": soft_dice_coef
    }

    model = tf.keras.models.load_model(model_filename, custom_objects=custom_objects)
    model.compile(
        optimizer='adam',
        loss=dice_coef_loss,
        metrics=['acc', combined_dice_ce_loss, dice_coef_loss, dice_coef, soft_dice_coef]
    )

    print("Evaluating model on test dataset. Please wait...")
    metrics = model.evaluate(
        x=X_test,
        y=Y_test,
        batch_size=8,
        verbose=1
    )

    for idx, metric in enumerate(metrics):
        print("Test dataset {} = {:.2f}".format(model.metrics_names[idx], metric))
