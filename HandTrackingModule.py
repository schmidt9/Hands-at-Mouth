import cv2
import mediapipe as mp
import time
import GeometryHelper


class HandDetector:

    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode,
                                         max_num_hands,
                                         model_complexity,
                                         min_detection_confidence,
                                         min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

        self.landmarkList = []

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, hand_index=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.landmarkList = []
        hull_points = []

        if self.results.multi_hand_landmarks:

            if hand_index > len(self.results.multi_hand_landmarks) - 1:
                return self.landmarkList, bbox, hull_points

            hand = self.results.multi_hand_landmarks[hand_index]
            points = []

            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)

                self.landmarkList.append([id, cx, cy])
                points.append([cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            hull_points = GeometryHelper.get_hull_points(points)

            if draw:
                cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20), (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
                GeometryHelper.plot_polylines(img, hull_points)

        return self.landmarkList, bbox, hull_points


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        detector.find_position(img)

        cTime = time.time()
        fps = 1. / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
