# Neuro Builder
import os
import logging
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import (
    Input, Conv2D, Activation, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def store_model(model):
    model_dir = 'Database/TrainedModel/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        log.info(f"Created directory: {model_dir}")

    model_json = model.to_json()
    with open(os.path.join(model_dir, "core.json"), "w") as model_file:
        model_file.write(model_json)
    model.save_weights(os.path.join(model_dir, "core.weights.h5"))

    log.info('Model architecture and weights have been saved.')


def build_model():
    inputs = Input(shape=(300, 300, 3))

    # First Convolutional Block
    conv1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(inputs)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)

    # Second Convolutional Block
    conv2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(act1)
    bn2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(bn2)

    # First Pooling Layer
    pool1 = MaxPooling2D(pool_size=(2, 2))(act2)
    drop1 = Dropout(0.25)(pool1)

    # Third Convolutional Block
    conv3 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(drop1)
    bn3 = BatchNormalization()(conv3)
    act3 = Activation('relu')(bn3)

    # Second Pooling Layer
    pool2 = MaxPooling2D(pool_size=(2, 2))(act3)
    drop2 = Dropout(0.25)(pool2)

    # Fourth Convolutional Block
    conv4 = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(drop2)
    bn4 = BatchNormalization()(conv4)
    act4 = Activation('relu')(bn4)

    # Third Pooling Layer
    pool3 = MaxPooling2D(pool_size=(2, 2))(act4)
    drop3 = Dropout(0.25)(pool3)

    # Flatten and Fully Connected Layers
    flat = Flatten()(drop3)
    dense1 = Dense(512)(flat)
    bn5 = BatchNormalization()(dense1)
    act5 = Activation('relu')(bn5)
    drop4 = Dropout(0.5)(act5)

    # Output Layer
    outputs = Dense(2, activation='softmax')(drop4)

    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model with a different optimizer
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )

    log.info("Model has been built and compiled.")
    return model


if __name__ == '__main__':
    log.info("Starting model build and storage process.")
    store_model(build_model())
    log.info("Model build and storage process completed.")
