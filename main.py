import cv2
import time

import GeometryHelper
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
    _, _, hand1_hull_points = handDetector.find_position(img, draw=True, hand_index=0)
    _, _, hand2_hull_points = handDetector.find_position(img, draw=True, hand_index=1)

    # lips

    img, lips_hull_points = lipsDetector.find_lips(img)

    # intersection

    hand1_intersects = GeometryHelper.points_intersect(hand1_hull_points, lips_hull_points)
    hand2_intersects = GeometryHelper.points_intersect(hand2_hull_points, lips_hull_points)

    if hand1_intersects or hand2_intersects:
        cv2.putText(img, "hand at mouth!", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    # fps

    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # result

    cv2.imshow("Image", img)
    cv2.waitKey(1)
