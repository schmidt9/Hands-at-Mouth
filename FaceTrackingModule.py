import cv2
import mediapipe as mp
import GeometryHelper


class FaceDetector:

    def __init__(self):
        self.mp_face_mash = mp.solutions.face_mesh
        self.face_mash = self.mp_face_mash.FaceMesh()
        self.face_mash_connection = self.mp_face_mash.FACEMESH_CONTOURS

    def find_face_mesh_connection(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mash.process(img_rgb)
        items = results.multi_face_landmarks

        hull_points = []

        if items:
            for landmarks in items:
                points = self.__get_face_mesh_connection_points(img, landmarks, self.face_mash_connection)
                # https://sefiks.com/2022/01/14/deep-face-detection-with-mediapipe/
                self.__plot_points(img, points)

                hull_points = GeometryHelper.get_hull_points(points)
                GeometryHelper.plot_polylines(img, hull_points)

        return img, hull_points

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


class IrisesDetector(FaceDetector):

    def __init__(self):
        super().__init__()
        self.face_mash_connection = self.mp_face_mash.FACEMESH_IRISES


class LipsDetector(FaceDetector):

    def __init__(self):
        super().__init__()
        self.face_mash_connection = self.mp_face_mash.FACEMESH_LIPS


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
