# Control Core
import time
import keyboard
import logging
from pynput.keyboard import Key, KeyCode
from pynput.mouse import Button, Controller as MouseCtrl

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

mouse_ctrl = MouseCtrl()

SPECIAL_KEY_ID_MAP = {
    Key.alt: 1001,
    Key.alt_l: 1002,
    Key.alt_r: 1003,
    Key.tab: 1004,
    Key.space: 1005,
    Key.enter: 1006,
    Key.esc: 1007,
    Key.up: 1008,
    Key.down: 1009,
    Key.left: 1010,
    Key.right: 1011,
}

ID_SPECIAL_KEY_MAP = {v: k for k, v in SPECIAL_KEY_ID_MAP.items()}

def press_key(key):
    try:
        keyboard.press(key)
        log.info(f"Pressed key: {key}")
    except Exception as e:
        log.error(f"Error pressing key {key}: {e}")

def release_key(key):
    try:
        keyboard.release(key)
        log.info(f"Released key: {key}")
    except Exception as e:
        log.error(f"Error releasing key {key}: {e}")

def perform_click(x, y):
    start_time = time.time()
    log.info(f"Executing click at position ({x}, {y}).")

    try:
        mouse_ctrl.position = (x, y)
        log.info(f"Cursor moved to position ({x}, {y}).")

        mouse_ctrl.press(Button.left)
        mouse_ctrl.release(Button.left)
        log.info("Mouse click executed.")
    except Exception as e:
        log.error(f"Error during mouse click: {e}")

    end_time = time.time()
    duration = end_time - start_time
    log.info(f"Click function completed. Time spent: {duration:.3f} seconds.")

def map_key_code(key_id):
    if key_id in ID_SPECIAL_KEY_MAP:
        log.info(f"Mapped key_id {key_id} to {ID_SPECIAL_KEY_MAP[key_id].name}.")
        return ID_SPECIAL_KEY_MAP[key_id].name
    else:
        try:
            mapped_key = chr(key_id)
            log.info(f"Mapped key_id {key_id} to {mapped_key}.")
            return mapped_key
        except Exception as err:
            log.error(f"Failed to map key_id {key_id}. Returning 'unknown', error {err}.")
            return 'unknown'

def get_id(key):
    if isinstance(key, Key):
        key_id = SPECIAL_KEY_ID_MAP.get(key, -1)
        log.info(f"Mapped special key {key} to ID {key_id}.")
        return key_id
    elif isinstance(key, KeyCode) and key.char is not None:
        key_id = ord(key.char.lower())
        log.info(f"Mapped KeyCode {key} to ASCII ID {key_id}.")
        return key_id
    else:
        log.error(f"Could not map key {key}. Returning -1.")
        return -1
