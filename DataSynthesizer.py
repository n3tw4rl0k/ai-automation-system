# Data Synthesizer
import os
import sys
import json
import argparse
import numpy as np
import logging
import pygetwindow as gw
from time import sleep, time
from ControlCore import get_id
from pynput.keyboard import Key
from PIL import ImageGrab, Image
from DataSmith import store_image
from multiprocessing import Process, Event
from pynput.mouse import Listener as MouseListener
from pynput.keyboard import Listener as KeyboardListener

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def capture_screen():
    img = ImageGrab.grab()
    img = img.convert('RGB')

    try:
        resample_method = Image.Resampling.LANCZOS
    except AttributeError:
        log.error("Please install Pillow version 10.x or newer.")
        sys.exit(1)

    img = img.resize((300, 300), resample=resample_method)
    img = np.array(img).astype('float32') / 255.0
    log.info("Screen captured and processed.")
    return img

def record_event(data_dir, event_type, event_info):
    timestamp = int(time())
    img_filename = f"{timestamp}.png"
    json_filename = f"{timestamp}.json"

    img_path = os.path.join(data_dir, img_filename)
    json_path = os.path.join(data_dir, json_filename)

    screenshot = capture_screen()
    store_image(screenshot, img_path)

    data = {
        "timestamp": timestamp,
        "mouse_event": event_info if event_type == "mouse" else None,
        "keyboard_event": event_info if event_type == "keyboard" else None
    }

    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    log.info(f"Event recorded: {event_type} -> {json_filename}.")

def record_keyboard_event(data_dir, event, key):
    if key in {Key.alt, Key.alt_l, Key.alt_r, Key.tab}:
        log.info(f"Excluded key event: {key}.")
        return

    key_id = get_id(key)
    if key_id == -1:
        log.error(f"Unexpected key: '{key}'. Event will not be saved.")
        return

    action = "press" if event == 1 else "release"

    event_data = {
        "action": action,
        "key_id": key_id
    }

    record_event(data_dir, "keyboard", event_data)

def record_mouse_event(data_dir, x, y):
    event_data = {
        "x": x,
        "y": y,
        "button": "left",
        "pressed": True
    }

    record_event(data_dir, "mouse", event_data)

def monitor_mouse(stop_evt, pause_evt, data_dir):
    os.makedirs(data_dir, exist_ok=True)

    def on_click(x, y, button, pressed):
        if pressed and not pause_evt.is_set():
            record_mouse_event(data_dir, x, y)

    with MouseListener(on_click=on_click) as listener:
        while not stop_evt.is_set():
            sleep(0.1)
        listener.stop()
        log.info("Mouse listener stopped.")

def monitor_keyboard(stop_evt, pause_evt, data_dir):
    os.makedirs(data_dir, exist_ok=True)

    def on_press(key):
        if not pause_evt.is_set():
            record_keyboard_event(data_dir, 1, key)

    def on_release(key):
        if not pause_evt.is_set():
            record_keyboard_event(data_dir, 2, key)

    with KeyboardListener(on_press=on_press, on_release=on_release) as listener:
        while not stop_evt.is_set():
            sleep(0.1)
        listener.stop()
        log.info("Keyboard listener stopped.")

def is_window_active(target_title):
    try:
        active_window = gw.getActiveWindow()
        if active_window:
            is_active = target_title.lower() in active_window.title.lower()
            log.info(f"Window '{target_title}' active status: {is_active}")
            return is_active
    except Exception as e:
        log.error(f"Error checking active window: {e}")
    return False

def monitor_window(target_title, stop_evt, pause_evt, check_interval=1):
    log.info(f"Waiting for window '{target_title}' to become active...")
    while not stop_evt.is_set():
        if is_window_active(target_title):
            if pause_evt.is_set():
                log.info(f"Window '{target_title}' is now active. Resuming capture.")
                pause_evt.clear()
        else:
            if not pause_evt.is_set():
                log.info(f"Window '{target_title}' is not active. Pausing capture.")
                pause_evt.set()
        sleep(check_interval)

def parse_args():
    parser = argparse.ArgumentParser(description="Monitor mouse and keyboard events for a specific game window.")
    parser.add_argument('--window-title', type=str, required=True, help='Name of the game window (e.g., Diablo.exe or Diablo)')
    return parser.parse_args()

def main():
    args = parse_args()
    window_title = args.window_title

    dataset_dir = os.path.join('Database', 'CollectedTrainingData')
    os.makedirs(dataset_dir, exist_ok=True)

    stop_evt = Event()
    pause_evt = Event()

    window_proc = Process(target=monitor_window, args=(window_title, stop_evt, pause_evt))
    window_proc.start()

    mouse_proc = Process(target=monitor_mouse, args=(stop_evt, pause_evt, dataset_dir))
    keyboard_proc = Process(target=monitor_keyboard, args=(stop_evt, pause_evt, dataset_dir))
    mouse_proc.start()
    keyboard_proc.start()

    try:
        window_proc.join()
        mouse_proc.join()
        keyboard_proc.join()
    except KeyboardInterrupt:
        log.info("CTRL+C detected. Stopping data capture...")
        stop_evt.set()
        window_proc.terminate()
        mouse_proc.terminate()
        keyboard_proc.terminate()
        window_proc.join()
        mouse_proc.join()
        keyboard_proc.join()
        log.info("Data capture stopped.")
    finally:
        stop_evt.set()
        window_proc.terminate()
        mouse_proc.terminate()
        keyboard_proc.terminate()
        window_proc.join()
        mouse_proc.join()
        keyboard_proc.join()
        log.info("Processes stopped successfully.")

if __name__ == '__main__':
    main()
