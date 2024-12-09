import numpy as np
import mediapipe as mp
import cv2
import pyautogui
from math import hypot
import time

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,        # specify whether an image is static or it is a video stream
    model_complexity=1,             # complexity of the hand landmark model: 0 or 1
    min_detection_confidence=0.75,  # minimum confidence, 0-1 default is 0.5
    min_tracking_confidence=0.75,   # minimum confidence, 0-1 default is 0.5
    max_num_hands=2)                # maximum number of hands default is 2

Draw = mp.solutions.drawing_utils

video = cv2.VideoCapture(2)

while True:
    success, frame = video.read()
    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)       # change to RGB

    landmarkList = []                                       # list of landmarks
    Process = hands.process(frameRGB)                       # process image to find hands

    # append landmarks to list and draw them on hand
    if Process.multi_hand_landmarks:
        for hand_lm in Process.multi_hand_landmarks:
            for _id, landmarks in enumerate(hand_lm.landmark):
                height, width, color_channel = frame.shape
                x, y = int(landmarks.x * width), int(landmarks.y * height)
                landmarkList.append([_id, x, y])

            Draw.draw_landmarks(frame, hand_lm, mpHands.HAND_CONNECTIONS)

    if landmarkList:
        x_thumb, y_thumb = landmarkList[4][1], landmarkList[4][2]           # coordinates of thumb tip
        x_index, y_index = landmarkList[8][1], landmarkList[8][2]           # coordinates of index tip
        x_middle, y_middle = landmarkList[12][1], landmarkList[12][2]       # coordinates of middle tip
        x_ring, y_ring = landmarkList[16][1], landmarkList[16][2]           # coordinates of ring tip

        cv2.circle(frame, (x_thumb, y_thumb), 7, (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, "cmd", (x_thumb-20, y_thumb-30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        cv2.circle(frame, (x_index, y_index), 7, (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, "copy", (x_index-20, y_index-30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        cv2.circle(frame, (x_middle, y_middle), 7, (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, "pase", (x_middle-20, y_middle-30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        cv2.circle(frame, (x_ring, y_ring), 7, (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, "quit", (x_ring-20, y_ring-30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)


        # interpolate distance between fingers to values between 0 and 100
        thumb_index = hypot(x_thumb-x_index, y_thumb-y_index)
        dist_thumb_index = np.interp(thumb_index, [15, 300], [0, 100])

        thumb_middle = hypot(x_thumb-x_middle, y_thumb-y_middle)
        dist_thumb_middle = np.interp(thumb_middle, [15, 300], [0, 100])

        thumb_ring = hypot(x_thumb-x_ring, y_thumb-y_ring)
        dist_thumb_ring = np.interp(thumb_ring, [15, 300], [0, 100])

        # find value for when tip is touching
        # print(int(dist_thumb_index))
        # print(int(dist_thumb_middle))
        # both less than 5

        # thumb+index is copy
        if int(dist_thumb_index) < 5:
            pyautogui.hotkey('command', 'c')
            time.sleep(0.25)

        # thumb+middle is paste
        if int(dist_thumb_middle) < 5:
            pyautogui.hotkey('command', 'v')
            time.sleep(0.25)

        if int(dist_thumb_ring) < 5:
            break

    # show webcam
    cv2.imshow("frame", frame)
    # stop webcam if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


