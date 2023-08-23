#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def scale_img(img, scale_percent=60):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


# Callback-Funktion für den linken Kamera-Topic
def left_image_callback(msg):
    global left_image
    left_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

# Callback-Funktion für den rechten Kamera-Topic
def right_image_callback(msg):
    global right_image
    right_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

# Callback-Funktion für die Trackbar-Änderungen (leere Funktion, kann nach Bedarf angepasst werden)
def nothing(x):
    pass

rospy.init_node('disparity_map_node')

# Initialisiere den Bildbrückenkonverter
bridge = CvBridge()

# Erzeuge die Fenster für das Disparity-Bild und die Trackbars
cv2.namedWindow('disp')
cv2.createTrackbar('numDisparities', 'disp', 1, 17, nothing)
cv2.createTrackbar('blockSize', 'disp', 5, 55, nothing)
cv2.createTrackbar('preFilterType', 'disp', 1, 1, nothing)
cv2.createTrackbar('preFilterSize', 'disp', 5, 55, nothing)
cv2.createTrackbar('preFilterCap', 'disp', 5, 62, nothing)
cv2.createTrackbar('textureThreshold', 'disp', 10, 100, nothing)
cv2.createTrackbar('uniquenessRatio', 'disp', 15, 100, nothing)
cv2.createTrackbar('speckleRange', 'disp', 0, 100, nothing)
cv2.createTrackbar('speckleWindowSize', 'disp', 3, 50, nothing)
cv2.createTrackbar('disp12MaxDiff', 'disp', 5, 25, nothing)
cv2.createTrackbar('minDisparity', 'disp', 5, 25, nothing)

# Variablen für die linken und rechten Bilder
left_image = None
right_image = None

# Abonniere die Kamera-Topics
rospy.Subscriber('/camera/stereo_left/image_rect', Image, left_image_callback)
rospy.Subscriber('/camera/stereo_right/image_rect', Image, right_image_callback)

while not rospy.is_shutdown():
    # Warte auf Bilder von beiden Kamera-Topics
    if left_image is not None and right_image is not None:
        # Konvertiere die Bilder in Graustufen

        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        left_gray_scaled = scale_img(left_gray)
        right_gray_scaled = scale_img(right_gray)
        # cv2.imwrite("left.png", left_image)
        # cv2.imwrite("right.png", right_image)


        stereo = cv2.StereoBM_create()

        # Updating the parameters based on the trackbar positions
        numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') * 16
        blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 5
        preFilterType = cv2.getTrackbarPos('preFilterType', 'disp')
        preFilterSize = cv2.getTrackbarPos('preFilterSize', 'disp') * 2 + 5
        preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
        textureThreshold = cv2.getTrackbarPos('textureThreshold', 'disp')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
        speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp') * 2
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
        minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')

        # Setting the updated parameters before computing disparity map
        stereo.setNumDisparities(numDisparities)
        stereo.setBlockSize(blockSize)
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setPreFilterCap(preFilterCap)
        stereo.setTextureThreshold(textureThreshold)
        stereo.setUniquenessRatio(uniquenessRatio)
        stereo.setSpeckleRange(speckleRange)
        stereo.setSpeckleWindowSize(speckleWindowSize)
        stereo.setDisp12MaxDiff(disp12MaxDiff)
        stereo.setMinDisparity(minDisparity)

        # Berechne das Disparity-Bild

        disparity = stereo.compute(left_gray_scaled, right_gray_scaled)

        # Converting to float32
        disparity = disparity.astype(np.float32)

        # Scaling down the disparity values and normalizing them
        disparity_img = None
        disparity_img = cv2.normalize(disparity, disparity_img, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        disparity_img = np.uint8(disparity_img)
        disparity_img = cv2.applyColorMap(disparity_img, cv2.COLORMAP_INFERNO)

        # disparity = (disparity / 16.0 - minDisparity) / numDisparities

        # Displaying the disparity map
        #disparity_resized = cv2.resize(disparity_img, (800, 700))
        # disparity_pseudocolor = cv2.applyColorMap(disparity_resized, cv2.COLORMAP_INFERNO)
        #cv2.imshow("disp", disparity_resized)
        cv2.imshow("disp", disparity_img)

        # Close window using esc key
        if cv2.waitKey(1) == 27 & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()