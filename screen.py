import cv2
import numpy as np
import pyautogui
import pygame
import win32gui
from PIL import Image


def capture_game(x, y, width, height):
    return capture_ss(x+10, y+70, width, height)

def capture_ss(left, top, width, height):
    capture_ss = pyautogui.screenshot(
        region=(int(left),
                int(top),
                int(width),
                int(height)))

    return resize(capture_ss)


def resize(capture_ss):
    # Preprocessing, resize to lower resolution
    # Maintain aspect ratio, 5x less data
    ratio = capture_ss.height / capture_ss.width
    new_width = int(capture_ss.width / 2)
    new_height = int(new_width * ratio)

    # Convert PIL Image to NP array
    arr_image = np.array(capture_ss)
    bgr = cv2.cvtColor(arr_image, cv2.COLOR_RGB2BGR)

    # Convert from RGB to BGR, then back to RGB
    resized_bgr = cv2.resize(bgr, (new_width, new_height), interpolation=cv2.INTER_NEAREST_EXACT)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    # resized_pil = Image.fromarray(resized_rgb)
    # resized_pil.show()

    return resized_rgb

