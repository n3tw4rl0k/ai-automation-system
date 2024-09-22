# Sentient Core
import os
import sys
import time
import argparse
import platform
import numpy as np
import logging
import pygetwindow as gw
from PIL import ImageGrab, Image
from keras.models import model_from_json
from ControlCore import press_key, release_key, perform_click, map_key_code

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def preprocess_image(screen):
    try:
        log.info("Preprocessing image...")
        img = Image.fromarray(screen)
        img = img.resize((300, 300))
        img = np.array(img)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        log.info("Image preprocessed.")
        return img
    except Exception as e:
        log.error(f"Error during preprocessing: {e}")
        return None

def make_prediction(model, screen):
    processed_img = preprocess_image(screen)
    if processed_img is None:
        log.error("Skipping prediction due to preprocessing error.")
        return None

    try:
        output = model.predict(processed_img)
        output = output[0]

        log.info(f"Model raw output: {output}")

        if len(output) == 2:
            mx, my = output
            kb_action = 0
            key_code = 0
        elif len(output) == 4:
            mx, my, kb_action, key_code = output
        else:
            log.error("Unexpected output shape from model.")
            mx, my, kb_action, key_code = 0, 0, 0, 0

        log.info(f"Model prediction: mx={mx}, my={my}, kb_action={kb_action}, key_code={key_code}")
        return [mx, my, kb_action, key_code]
    except Exception as e:
        log.error(f"Error during prediction: {e}")
        return None

class AutoController:
    def __init__(self, title, json_path, weights_path):
        self.title = title
        self.json_path = json_path
        self.weights_path = weights_path
        self.model = self.load_model()
        self.window = None

    def load_model(self):
        try:
            log.info("Loading model...")
            with open(self.json_path, 'r') as file:
                model_json = file.read()
            mdl = model_from_json(model_json)
            mdl.load_weights(self.weights_path)
            log.info('Model loaded successfully.')
            return mdl
        except Exception as e:
            log.error(f"Error loading model: {e}")
            sys.exit(1)

    def get_window(self):
        try:
            windows = gw.getWindowsWithTitle(self.title)
            if windows:
                return windows[0]
            return None
        except Exception as e:
            log.error(f"Error retrieving window: {e}")
            return None

    def is_active(self):
        try:
            active_win = gw.getActiveWindow()
            is_act = active_win and active_win.title == self.title
            return is_act
        except Exception as e:
            log.error(f"Error checking active window: {e}")
            return False

    def capture_window_screen(self, window):
        try:
            log.info("Capturing screen...")
            bbox = (
                window.left,
                window.top,
                window.right,
                window.bottom
            )
            img = ImageGrab.grab(bbox=bbox)
            img_np = np.array(img)
            log.info("Screen captured.")
            return img_np
        except Exception as e:
            log.error(f"Error capturing screen: {e}")
            return None

    def execute_actions(self, actions):
        if actions is None:
            log.info("No actions to execute.")
            return

        mx, my, kb_action, key_code = actions
        log.info(f"Executing actions: mx={mx}, my={my}, kb_action={kb_action}, key_code={key_code}")

        if mx == -1 and my == -1:
            key = map_key_code(key_code)
            if kb_action == 1:
                try:
                    log.info(f"Pressing key: {key}")
                    press_key(key)
                except Exception as e:
                    log.error(f"Error pressing key {key}: {e}")
            elif kb_action == 0:
                try:
                    log.info(f"Releasing key: {key}")
                    release_key(key)
                except Exception as e:
                    log.error(f"Error releasing key {key}: {e}")
        elif kb_action == 0 and key_code == 0:
            try:
                log.info(f"Performing mouse click at ({mx}, {my})")
                self.perform_click(mx, my)
            except Exception as e:
                log.error(f"Error performing mouse click at ({mx}, {my}): {e}")
        else:
            try:
                log.info(f"Performing mouse click at ({mx}, {my}) and keyboard action")
                self.perform_click(mx, my)
                key = map_key_code(key_code)
                if kb_action == 1:
                    log.info(f"Pressing key: {key}")
                    press_key(key)
                elif kb_action == 0:
                    log.info(f"Releasing key: {key}")
                    release_key(key)
            except Exception as e:
                log.error(f"Error performing combined actions: {e}")

    def perform_click(self, x, y):
        try:
            if not self.window:
                log.error("Window not set. Cannot perform click.")
                return

            abs_x = int(x * self.window.width) + self.window.left
            abs_y = int(y * self.window.height) + self.window.top
            log.info(f"Clicking at absolute position ({abs_x}, {abs_y})")
            perform_click(abs_x, abs_y)
        except Exception as e:
            log.error(f"Error performing click at ({x}, {y}): {e}")

    def run(self):
        log.info("AI Controller running...")
        while True:
            try:
                if self.is_active():
                    log.info(f"Window '{self.title}' is active.")
                    window = self.get_window()
                    if window is None:
                        log.error(f"Window '{self.title}' not found.")
                        time.sleep(0.5)
                        continue

                    self.window = window

                    screen = self.capture_window_screen(window)
                    if screen is not None:
                        pred = make_prediction(self.model, screen)
                        self.execute_actions(pred)
                else:
                    log.info(f"Window '{self.title}' is not active.")

                time.sleep(0.5)
            except KeyboardInterrupt:
                log.info("AI Controller stopped by user.")
                sys.exit(0)
            except Exception as e:
                log.error(f"Unhandled error occurred: {e}")
                time.sleep(0.5)
                continue

def parse_args():
    parser = argparse.ArgumentParser(description='AI Automation Script')

    parser.add_argument(
        '--window_title',
        type=str,
        required=True,
        help='Title of the target window'
    )

    default_model_dir = os.path.join('Database', 'TrainedModel')
    default_model_json = os.path.join(default_model_dir, 'core.json')
    default_model_weights = os.path.join(default_model_dir, 'core.weights.h5')

    parser.add_argument(
        '--model_json',
        type=str,
        default=default_model_json,
        help=f'Path to the model JSON file (default: {default_model_json})'
    )
    parser.add_argument(
        '--model_weights',
        type=str,
        default=default_model_weights,
        help=f'Path to the model weights file (default: {default_model_weights})'
    )

    args = parser.parse_args()
    return args.window_title, args.model_json, args.model_weights

def main():
    if platform.system() != 'Windows':
        log.error("This script only runs on Windows.")
        sys.exit(1)

    window_title, model_json, model_weights = parse_args()

    if not os.path.exists(model_json):
        log.error(f"Model JSON file not found at path: {model_json}")
        sys.exit(1)
    if not os.path.exists(model_weights):
        log.error(f"Model weights file not found at path: {model_weights}")
        sys.exit(1)

    ai_ctrl = AutoController(window_title, model_json, model_weights)
    log.info("AI Controller started.")
    ai_ctrl.run()

if __name__ == '__main__':
    main()
