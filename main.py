import cv2
import time
import sys
import argparse

import GeometryUtils
import HandTrackingModule
import FaceTrackingModule
from HandsAtMouthHandler import HandsAtMouthHandler
from WindowMinimizer import WindowMinimizer

# process options

opts = [opt for opt in sys.argv[1:] if opt.startswith("--")]
values = [opt for opt in sys.argv[1:] if not opt.startswith("--")]
window_size = (640, 480)
window_position = (400, 400)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--no-gui", help="Start in windowless (no GUI) mode", action='store_true')
arg_parser.add_argument("--topmost", help="If in GUI mode show window on top", action='store_true')
arg_parser.add_argument("--window-size",
                        help=f"Window size specified as tuple (width, height). Works in GUI mode",
                        default=f"({window_size[0]},{window_size[1]})")
arg_parser.add_argument("--window-position",
                        help="Window position specified as tuple (x, y). Works in GUI mode",
                        default=f"({window_position[0]},{window_position[1]})")

args = arg_parser.parse_args()

no_gui = args.no_gui
topmost = args.topmost
window_size = eval(args.window_size)
window_position = eval(args.window_position)
with_gui = not no_gui

print("Staring in windowless mode" if no_gui else "Starting in GUI mode")

if with_gui and topmost:
    print("Starting in topmost mode")

print(f"Window size {window_size}, position {window_position}")

# setup

window_name = "Hand At Mouth"
capture = cv2.VideoCapture(0)

if with_gui:
    capture.set(3, window_size[0])
    capture.set(4, window_size[1])

pTime = 0

handDetector = HandTrackingModule.HandDetector(min_detection_confidence=0.75)
lipsDetector = FaceTrackingModule.LipsDetector()

handler = HandsAtMouthHandler()
handler.add_listener(WindowMinimizer("Google Chrome"))

# run

while True:
    success, img = capture.read()

    if with_gui:
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
        print("Hands At Mouth!")

        handler.handle_hands_at_mouth()

        if with_gui:
            cv2.putText(img, "hand at mouth!", (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    if with_gui:
        # fps

        cTime = time.time()
        fps = 1 / (cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # result

        cv2.imshow(window_name, img)

        if topmost:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        cv2.moveWindow(window_name, window_position[0], window_position[1])

    cv2.waitKey(1)
    cv2.destroyAllWindows()
