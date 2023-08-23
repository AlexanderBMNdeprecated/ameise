import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

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

# Variablen für die linken und rechten Bilder
left_image = None
right_image = None

orb = cv2.ORB_create(nfeatures=1000)

# Abonniere die Kamera-Topics
rospy.Subscriber('/camera/stereo_left/image_rect', Image, left_image_callback)
rospy.Subscriber('/camera/stereo_right/image_rect', Image, right_image_callback)


while not rospy.is_shutdown():
    if left_image is not None and right_image is not None:

        left_image = cv2.imread('old_toDelete/left.png')
        right_image = cv2.imread('old_toDelete/right.png')

        orb_keypoints_left = orb.detect(left_image, None)
        orb_keypoints_left, orb_descriptors_left = orb.compute(left_image, orb_keypoints_left)
        cv2_image_orb_left = cv2.drawKeypoints(left_image, orb_keypoints_left, None, color=(0, 255, 0), flags=0)

        #cv2.imshow("test", cv2_image_orb_left)


        orb_keypoints_right = orb.detect(right_image, None)
        orb_keypoints_right, orb_descriptors_right = orb.compute(right_image, orb_keypoints_right)
        cv2_image_orb_right = cv2.drawKeypoints(right_image, orb_keypoints_right, None, color=(0, 255, 0), flags=0)

        #cv2.imshow("test", cv2_image_orb_right)

        matcher = cv2.BFMatcher()
        matches = matcher.match(orb_descriptors_left, orb_descriptors_right)
        final_img = cv2.drawMatches(left_image, orb_keypoints_left,
                                    right_image, orb_descriptors_right,
                                    matches, None)
        #final_img = cv2.putText(final_img, str(len(matches)), (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA, False)
        cv2.imshow("Matches", final_img)



        # Close window using esc key
        if cv2.waitKey(1) == 27 & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()


