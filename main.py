import cv2
import time

import HandTrackingModule
import LipsTrackingModule

wCam, hCam = 640, 480

capture = cv2.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)

pTime = 0

handDetector = HandTrackingModule.HandDetector(detectionCon=0.75)
lipsDetector = LipsTrackingModule.LipsDetector()
totalFingers = 0

while True:
    success, img = capture.read()
    img = cv2.flip(img, 1)

    # hands

    img = handDetector.find_hands(img)
    lmList, bbox, hand_hull_points = handDetector.find_position(img, draw=True)

    if lmList:
        fingersUp = handDetector.fingers_up()
        totalFingers = fingersUp.count(1)

    cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    # lips

    img, lips_hull_points = lipsDetector.find_lips(img)

    # fps

    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # result

    cv2.imshow("Image", img)
    cv2.waitKey(1)
