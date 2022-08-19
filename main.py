import cv2
import time

import GeometryUtils
import HandTrackingModule
import FaceTrackingModule
from HandsAtMouthHandler import HandsAtMouthHandler
from WindowMinimizer import WindowMinimizer

wCam, hCam = 640, 480

capture = cv2.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)

pTime = 0

handDetector = HandTrackingModule.HandDetector(min_detection_confidence=0.75)
lipsDetector = FaceTrackingModule.LipsDetector()

handler = HandsAtMouthHandler()
handler.add_listener(WindowMinimizer("Google Chrome"))

while True:
    success, img = capture.read()
    img = cv2.flip(img, 1)

    # hands

    img = handDetector.find_hands(img)
    _, _, hand1_hull_points = handDetector.find_position(img, draw=True, hand_index=0)
    _, _, hand2_hull_points = handDetector.find_position(img, draw=True, hand_index=1)

    # lips

    img, lips_hull_points = lipsDetector.find_face_mesh_connection(img)

    # intersection

    hand1_intersects = GeometryUtils.points_intersect(hand1_hull_points, lips_hull_points)
    hand2_intersects = GeometryUtils.points_intersect(hand2_hull_points, lips_hull_points)

    is_hand_at_mouth = hand1_intersects or hand2_intersects

    if is_hand_at_mouth:
        cv2.putText(img, "hand at mouth!", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        handler.handle_hands_at_mouth()

    # fps

    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # result

    cv2.imshow("Image", img)
    cv2.waitKey(1)
