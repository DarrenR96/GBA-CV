import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import pyautogui

def padToShape(a, shape):
    y_, x_ = shape
    y, x, _ = a.shape
    y_pad = (y_-y)
    x_pad = (x_-x)
    return np.pad(a,((y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2), (0, 0)),
                  mode = 'constant'), x_pad//2, y_pad//2
pyautogui.PAUSE = 0.01
controller = 'data/Controller.png'
controller = np.asarray(Image.open(controller))[::2, ::2, :]
cap = cv2.VideoCapture(1)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(controller.shape)
controller, x_offset, y_offset = padToShape(controller, (height, width))

buttons = {
    'a' : [(48, 137),(87, 175)],
    'd' : [(125, 135),(165, 175)],
    'w': [(87,97),(124,137)],
    's' : [(87,175),(124,216)],
    'backspace' : [(225,170),(280,205)],
    'enter' : [(310,170),(367,205)],
    'k' : [(413,150),(487,223)],
    'l': [(494,150),(568,223)]
}

for _key in buttons.keys():
    upX, upY = buttons[_key][0]
    downX, downY = buttons[_key][1]
    controller = cv2.rectangle(controller, (x_offset+upX, y_offset+upY), (x_offset+downX, y_offset+downY), (0, 0, 206), 5)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, image = cap.read()
    image = cv2.flip(image, 1)
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    controllerBGRA = cv2.cvtColor(controller, cv2.COLOR_RGB2BGRA)
    results = hands.process(imageRGB)

   # checking whether a hand is detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: # working with each hand 
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if (id == 8):
                    cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    for _key in buttons.keys():
                        upX, upY = buttons[_key][0]
                        downX, downY = buttons[_key][1]
                        if ((upX + x_offset) <= cx <= (downX + x_offset)) and ((upY + y_offset) <= cy <= (downY + y_offset)):
                            controller = cv2.rectangle(controller, (x_offset+upX, y_offset+upY), (x_offset+downX, y_offset+downY), (0, 255, 0), 5)
                            pyautogui.press(_key, interval=0.01)
                        else:
                            controller = cv2.rectangle(controller, (x_offset+upX, y_offset+upY), (x_offset+downX, y_offset+downY), (0, 0, 206), 5)

            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
    
    image = cv2.cvtColor(image,cv2.COLOR_BGR2BGRA)
    image = cv2.addWeighted(image,0.7,controllerBGRA,0.7,0)

    cv2.imshow("Output", image)
    cv2.waitKey(1)