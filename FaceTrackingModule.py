import math
import cv2
import mediapipe as mp
import cvzone
import GeometryUtils
import ImageUtils


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


class IrisesDetector(FaceDetector):

    def __init__(self):
        # we set refine_landmarks=True here
        # to avoid 'index out of bounds error' in __get_face_mesh_connection_points
        super().__init__(refine_landmarks=True)
        self.pixel_sunglasses_image = self.__class__.__read_pixel_sunglasses_image()
        self.left_iris_hull_points = []
        self.right_iris_hull_points = []

    def find_face_mesh_connection(self, img):
        self.face_mash_connection = self.mp_face_mash.FACEMESH_LEFT_IRIS
        _, self.left_iris_hull_points = super().find_face_mesh_connection(img)

        self.face_mash_connection = self.mp_face_mash.FACEMESH_RIGHT_IRIS
        _, self.right_iris_hull_points = super().find_face_mesh_connection(img)

        return img, self.hull_points

    def add_pixel_sunglasses_image(self, img):

        # resize image to fit eyes area

        width_scale_factor = 2.0
        height_scale_factor = 2.0

        left_iris_center = GeometryUtils.centroid(self.left_iris_hull_points)
        right_iris_center = GeometryUtils.centroid(self.right_iris_hull_points)

        if left_iris_center.is_empty or right_iris_center.is_empty:
            return img

        irises_distance = abs(left_iris_center.x - right_iris_center.x)
        # using enclosing size here to get the same height on head tilt
        irises_height = GeometryUtils.min_enclosing_circle_size(self.right_iris_hull_points)

        new_image_width = int(irises_distance * width_scale_factor)
        new_image_height = int(irises_height * height_scale_factor)

        resized_image = cv2.resize(self.pixel_sunglasses_image,
                                   (new_image_width, new_image_height),
                                   interpolation=cv2.INTER_AREA)

        # position image based on right iris position

        x = int(right_iris_center.x - ((new_image_width - irises_distance) / 2))
        y = int(right_iris_center.y - (new_image_height / 2))

        # rotate image around right iris on head tilt

        tilt_degrees = math.degrees(math.atan2(
            left_iris_center.x - right_iris_center.x,
            left_iris_center.y - right_iris_center.y) - math.pi / 2)
        print(tilt_degrees)

        center = (0, 0)
        resized_image = ImageUtils.rotate_image(resized_image, tilt_degrees, center)

        # combine images

        # img = ImageUtils.add_transparent_image(img, resized_image, x, y, tilt_degrees, center)
        img = cvzone.overlayPNG(img, resized_image, [x, y])

        return img

    @staticmethod
    def __read_pixel_sunglasses_image():
        image_name = "pixel_sunglasses.png"
        image_dir = "img"
        # using IMREAD_UNCHANGED to read image with alpha channel
        return cv2.imread(f'{image_dir}/{image_name}', cv2.IMREAD_UNCHANGED)


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
    irises_detector = IrisesDetector()

    while True:
        success, img = capture.read()
        img = cv2.flip(img, 1)
        img, _ = lips_detector.find_face_mesh_connection(img)
        img, _ = irises_detector.find_face_mesh_connection(img)
        img = irises_detector.add_pixel_sunglasses_image(img)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
