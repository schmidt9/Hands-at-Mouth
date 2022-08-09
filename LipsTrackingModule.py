import cv2
import mediapipe as mp


class LipsDetector:

    def __init__(self):
        self.mpFace = mp.solutions.face_mesh
        self.faceMash = self.mpFace.FaceMesh()
        self.mpDraw = mp.solutions.drawing_utils

    @staticmethod
    def plot_landmark(img, landmarks, facial_area_obj):
        for source_idx, target_idx in facial_area_obj:
            source = landmarks.landmark[source_idx]
            target = landmarks.landmark[target_idx]

            relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
            relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))

            cv2.line(img, relative_source, relative_target, (255, 255, 255), thickness=2)

    def find_face(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMash.process(imgRGB)
        items = results.multi_face_landmarks

        if items:
            for landmarks in items:
                # https://sefiks.com/2022/01/14/deep-face-detection-with-mediapipe/
                self.plot_landmark(img, landmarks, self.mpFace.FACEMESH_LIPS)

        return img


def main():
    cap = cv2.VideoCapture(0)
    detector = LipsDetector()

    while True:
        success, img = cap.read()
        img = detector.find_face(img)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
