# Vision Predictor
import numpy as np
import logging
from skimage.transform import resize

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def make_prediction(model, X):
    X_resized = resize(X, (150, 150, 3), anti_aliasing=True)
    log.info("Image resized to 150x150.")

    X_resized = X_resized.astype('float32') / 255.0
    log.info("Image normalized.")

    if X_resized.ndim == 2:
        X_resized = np.expand_dims(X_resized, axis=-1)
        log.info("Expanded grayscale image dimensions.")
    elif X_resized.shape[2] == 4:
        X_resized = X_resized[..., :3]
        log.info("Converted RGBA image to RGB.")

    X_prepared = X_resized.reshape(1, 150, 150, 3)
    log.info("Image reshaped for model input.")

    try:
        Y = model.predict(X_prepared)
        log.info("Prediction made successfully.")
        return Y
    except Exception as e:
        log.error(f"Error during prediction: {e}")
        return None
