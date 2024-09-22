# Data Smith
import os
import json
import logging
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def load_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        resample_method = Image.Resampling.LANCZOS
        img = img.resize((300, 300), resample_method)
        img_array = np.array(img).astype('float32') / 255.0
        log.info(f"Successfully loaded and processed image: {image_path}")
        return img_array
    except Exception as e:
        log.error(f"Error loading image {image_path}: {e}")
        return None

def store_image(img, path):
    try:
        img = np.clip(img, 0, 1)
        img_uint8 = (img * 255).astype('uint8')
        image = Image.fromarray(img_uint8)
        image.save(path)
        log.info(f"Successfully saved image to: {path}")
    except Exception as e:
        log.error(f"Error saving image {path}: {e}")

def prepare_dataset(dataset_dir='Database/CollectedTrainingData'):
    npy_train_dir = 'Database/NumpyTrainData'
    X_npy = os.path.join(npy_train_dir, 'X.npy')
    Y_npy = os.path.join(npy_train_dir, 'Y.npy')

    if os.path.exists(X_npy) and os.path.exists(Y_npy):
        try:
            X = np.load(X_npy)
            Y = np.load(Y_npy)
            log.info("Loaded existing dataset from .npy files.")
        except Exception as e:
            log.error(f"Error loading .npy files: {e}")
            X, Y = None, None
    else:
        X, Y = None, None

    if X is None or Y is None:
        X = []
        Y = []
        label_mapping = {"mouse": 0, "keyboard": 1}

        log.info(f"Processing data from {dataset_dir}...")

        for filename in os.listdir(dataset_dir):
            if filename.endswith('.json'):
                json_path = os.path.join(dataset_dir, filename)
                with open(json_path, 'r') as f:
                    metadata = json.load(f)

                img_filename = filename.replace('.json', '.png')
                img_path = os.path.join(dataset_dir, img_filename)
                img = load_image(img_path)
                if img is not None:
                    if metadata["mouse_event"] is not None:
                        label = "mouse"
                    elif metadata["keyboard_event"] is not None:
                        label = "keyboard"
                    else:
                        label = "unknown"

                    if label != "unknown":
                        X.append(img)
                        Y.append(label_mapping[label])

        X = np.array(X).astype('float32')
        Y = np.array(Y).astype('float32')

        Y = to_categorical(Y, num_classes=len(label_mapping))

        if not os.path.exists(npy_train_dir):
            os.makedirs(npy_train_dir)

        try:
            np.save(X_npy, X)
            np.save(Y_npy, Y)
            log.info(f"Saved dataset to {X_npy} and {Y_npy}.")
        except Exception as e:
            log.error(f"Error saving .npy files: {e}")

    if len(X) == 0 or len(Y) == 0:
        log.error("Dataset is empty. Ensure images are loaded correctly.")
        raise ValueError("Dataset is empty. Ensure images are loaded correctly.")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42, shuffle=True
    )

    log.info(f"Dataset split into {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples.")
    return X_train, X_test, Y_train, Y_test
