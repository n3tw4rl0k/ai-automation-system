# Skill Forge
import os
import h5py
import math
import logging
import tensorflow as tf
from DataSmith import prepare_dataset
from NeuroBuilder import build_model, store_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

log.info(f"TensorFlow version: {tf.__version__}")
log.info(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
log.info(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
log.info(f"h5py version: {h5py.__version__}")

epochs = 200
batch_size = 5


def train_ai_model(model, X, X_test, Y, Y_test):
    callbacks = []
    checkpoints_dir = 'Database/Checkpoints/'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        log.info(f"Created checkpoints directory: {checkpoints_dir}")

    callbacks.append(ModelCheckpoint(
        os.path.join(checkpoints_dir, 'best_weights.weights.h5'),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        save_freq='epoch'
    ))

    callbacks.append(TensorBoard(
        log_dir=os.path.join(checkpoints_dir, 'logs'),
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        embeddings_freq=0
    ))

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    callbacks.append(early_stopping)

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    datagen.fit(X)
    log.info("Data augmentation setup complete.")

    steps_per_epoch = math.ceil(len(X) / batch_size)

    log.info(f"Starting model training for {epochs} epochs.")
    model.fit(
        datagen.flow(X, Y, batch_size=batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(X_test, Y_test),
        shuffle=True,
        callbacks=callbacks
    )
    log.info("Model training completed.")

    return model


def main():
    X, X_test, Y, Y_test = prepare_dataset()
    log.info("Dataset loaded.")
    model = build_model()
    log.info("Model built.")
    model = train_ai_model(model, X, X_test, Y, Y_test)
    store_model(model)
    log.info("Model stored.")
    return model


if __name__ == '__main__':
    main()
