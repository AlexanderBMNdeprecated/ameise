#!/home/ameise/catkin_ws/src/venv/bin python

from __future__ import print_function
import cv2
from cv_bridge import CvBridge, CvBridgeError


class ImageStitcher:
    def __init__(self):
        self.bridge = CvBridge()
        self.stitcher = cv2.Stitcher.create()
        self.images = []
    
    def callback_image(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)  # rotieren um 180 Grad
            self.images.append(cv_image)
        except CvBridgeError as e:
            print(e)

    def clean_images(self):
        self.images = []

    def get_panorama(self):
        if self.images:
            status, result = self.stitcher.stitch(self.images)
            if status != cv2.STITCHER_OK:
                print(status)
                print("There was an error in the stitching procedure")
                return -1
            else:
                # return pano
                self.clean_images()
                return result
