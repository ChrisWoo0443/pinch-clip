import numpy as np
import mediapipe as mp
import cv2
import pyautogui
from math import hypot

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,        # specify whether an image is static or it is a video stream
    model_complexity=1,             # complexity of the hand landmark model: 0 or 1
    min_detection_confidence=0.75,  # minimum confidence, 0-1 default is 0.5
    min_tracking_confidence=0.75,   # minimum confidence, 0-1 default is 0.5
    max_num_hands=2)                # maximum number of hands default is 2

Draw = mp.solutions.drawing_utils

video = cv2.VideoCapture(0)

while True:
    success, frame = video.read()
    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)       # change to RGB

    landmarkList = []                                       # list of landmarks
    Process = hands.process(frameRGB)                       # process image to find hands

    # append landmarks to list and draw them on hand
    if Process.multi_hand_landmarks:
        for handlm in Process.multi_hand_landmarks:
            for _id, landmarks in enumerate(handlm.landmark):
                height, width, color_channel = frame.shape
                x, y = int(landmarks.x * width), int(landmarks.y * height)
                landmarkList.append([_id, x, y])

            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)



    # show webcam
    cv2.imshow("frame", frame)
    # stop webcam if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


