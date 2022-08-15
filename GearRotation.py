import cvzone
import cv2
import numpy as np
import ImageUtils

angle = 0
fpsReader = cvzone.FPS()

while True:
    imgBack = np.ones((1000, 1000, 3), np.uint8) * 255
    imgG1 = cv2.imread("img/pixel_sunglasses.png", cv2.IMREAD_UNCHANGED)

    imgG1 = cv2.resize(imgG1,
                       (200, 80),
                       interpolation=cv2.INTER_AREA)

    imgG1 = ImageUtils.rotate_image(imgG1, angle)
    angle += 1

    imgResult = cvzone.overlayPNG(imgBack, imgG1, [125, 300])
    _, imgResult = fpsReader.update(imgResult)

    cv2.imshow("Image", imgResult)
    cv2.waitKey(1)
