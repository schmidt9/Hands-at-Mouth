import cv2
import mediapipe as mp
import GeometryHelper


class LipsDetector:

    def __init__(self):
        self.mp_face_mash = mp.solutions.face_mesh
        self.face_mash = self.mp_face_mash.FaceMesh()

    def find_lips(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mash.process(imgRGB)
        items = results.multi_face_landmarks

        hull_points = []

        if items:
            for landmarks in items:
                points = self.__get_lips_points(img, landmarks, self.mp_face_mash.FACEMESH_LIPS)
                # https://sefiks.com/2022/01/14/deep-face-detection-with-mediapipe/
                self.__plot_all_points(img, points)

                hull_points = GeometryHelper.get_hull_points(points)
                GeometryHelper.plot_polylines(img, hull_points)

        return img, hull_points

    @staticmethod
    def __get_lips_points(img, face_landmarks, facial_area_obj):
        points = []

        for source_idx, target_idx in facial_area_obj:
            source = face_landmarks.landmark[source_idx]
            target = face_landmarks.landmark[target_idx]

            relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
            relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))

            points.append(relative_source)
            points.append(relative_target)

        return points

    @staticmethod
    def __plot_all_points(img, points):
        for i in range(0, len(points), 2):
            cv2.line(img, points[i], points[i + 1], (255, 255, 255), thickness=2)


def main():
    cap = cv2.VideoCapture(0)
    detector = LipsDetector()

    while True:
        success, img = cap.read()
        img = detector.find_lips(img)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
