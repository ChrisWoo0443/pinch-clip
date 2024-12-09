import numpy as np
import mediapipe as mp
import cv2
import pyperclip
from math import hypot

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,        # specify whether an image is static or it is a video stream
    model_complexity=1,             # complexity of the hand landmark model: 0 or 1
    min_detection_confidence=0.75,  # minimum confidence, 0-1 default is 0.5
    min_tracking_confidence=0.75,   # minimum confidence, 0-1 default is 0.5
    max_num_hands=2)                # maximum number of hands default is 2

Draw = mp.solutions.drawing_utils

