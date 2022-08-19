import cv2
import mediapipe as mp
import GeometryUtils


class FaceDetector:

    def __init__(self, refine_landmarks=False):
        self.mp_face_mash = mp.solutions.face_mesh
        self.face_mash = self.mp_face_mash.FaceMesh(refine_landmarks=refine_landmarks)
        self.face_mash_connection = self.mp_face_mash.FACEMESH_CONTOURS
        self.hull_points = []

    def find_face_mesh_connection(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mash.process(img_rgb)
        items = results.multi_face_landmarks

        self.hull_points = []

        if items:
            for landmarks in items:
                points = self.__get_face_mesh_connection_points(img, landmarks, self.face_mash_connection)
                # https://sefiks.com/2022/01/14/deep-face-detection-with-mediapipe/
                self.__plot_points(img, points)

                self.hull_points = GeometryUtils.get_hull_points(points)
                GeometryUtils.plot_polylines(img, self.hull_points)

        return img, self.hull_points

    @staticmethod
    def __get_face_mesh_connection_points(img, face_landmarks, face_mesh_connection):
        points = []

        for source_idx, target_idx in face_mesh_connection:
            source = face_landmarks.landmark[source_idx]
            target = face_landmarks.landmark[target_idx]

            relative_source = (int(img.shape[1] * source.x), int(img.shape[0] * source.y))
            relative_target = (int(img.shape[1] * target.x), int(img.shape[0] * target.y))

            points.append(relative_source)
            points.append(relative_target)

        return points

    @staticmethod
    def __plot_points(img, points):
        for i in range(0, len(points), 2):
            cv2.line(img, points[i], points[i + 1], (255, 255, 255), thickness=2)


class LipsDetector(FaceDetector):

    def __init__(self):
        super().__init__()
        self.face_mash_connection = self.mp_face_mash.FACEMESH_LIPS


def main():
    wCam, hCam = 640, 480
    capture = cv2.VideoCapture(0)
    capture.set(3, wCam)
    capture.set(4, hCam)

    lips_detector = LipsDetector()

    while True:
        success, img = capture.read()
        img = cv2.flip(img, 1)
        img, _ = lips_detector.find_face_mesh_connection(img)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
