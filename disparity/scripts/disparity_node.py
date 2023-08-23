#!/usr/bin/env python3

import sys
import cv2
import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from stereo_msgs.msg import DisparityImage
import std_msgs.msg
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
import numpy as np
import time
import os
import yaml
import struct
import message_filters
import datetime
import rospkg


class CamInfo:
    def __init__(self, cam_info_msg):
        """
        Class to store der camera information
        Args:
            cam_info_msg:
        """
        self.header = cam_info_msg.header
        self.header.frame_id = "map"
        self.frame_id = cam_info_msg.header.frame_id
        # Extract height and width
        self.height = cam_info_msg.height
        self.width = cam_info_msg.width
        # Extrahiere die Kameramatrix und Verzerrungskoeffizienten
        self.cam_matrix = np.array(cam_info_msg.K).reshape(3, 3)
        self.dist_coeff = np.array(cam_info_msg.D)
        # Extrahiere die Rektifizierungsmatrix und die Projektionsmatrix
        self.rect_matrix = np.array(cam_info_msg.R).reshape(3, 3)
        self.proj_matrix = np.array(cam_info_msg.P).reshape(3, 4)
        # Extrahiere das Region of Interest (ROI)-Rechteck
        self.roi = cam_info_msg.roi


class StereoConfig:
    def __init__(self, config_file):
        self.numDisparities = config_file['numDisparities']
        self.blockSize = config_file['blockSize']
        self.preFilterType = config_file['preFilterType']
        self.preFilterSize = config_file['preFilterSize']
        self.preFilterCap = config_file['preFilterCap']
        self.textureThreshold = config_file['textureThreshold']
        self.uniquenessRatio = config_file['uniquenessRatio']
        self.speckleRange = config_file['speckleRange']
        self.speckleWindowSize = config_file['speckleWindowSize']
        self.disp12MaxDiff = config_file['disp12MaxDiff']
        self.minDisparity = config_file['minDisparity']
        transformation = config_file["T_cn_cnm1"]
        self.rotation = np.array(transformation[:3])[:, :3]
        self.translation = np.array(transformation[:3])[:, 3]
        self.baseline = np.array(transformation[0])[3]
        self.focal_length = np.array(transformation[0])[0]

_FIELDS_XYZ = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
]
_FIELDS_XYZRGB = _FIELDS_XYZ + [PointField(name="rgba", offset=12, datatype=PointField.UINT32, count=1)]

def xyzrgb_array_to_pointcloud2(projected_points, colors, max_depth=40.0, header=None):
    points_3d = []
    #projected_points.tolist()

    if len(colors.shape) == 2:
        colors = np.dstack([colors, colors, colors])
    height, width = colors.shape[:2]
    for u in range(height):
        img_row = colors[u]
        projected_row = projected_points[u]
        for v in range(width):
            cur_projected_point = projected_row[v]
            x = cur_projected_point[2]
            y = -cur_projected_point[0]
            z = -cur_projected_point[1]
            if x < 0 or x > max_depth:
                continue
            cur_color = img_row[v]
            b = cur_color[0]
            g = cur_color[1]
            r = cur_color[2]
            a = 255
            rgb = struct.unpack("I", struct.pack("BBBB", b, g, r, a))[0]
            pt = [x, y, z, rgb]
            points_3d.append(pt)

    if header is None:
        pointcloud_msg = pc2.create_cloud(std_msgs.msg.Header(), _FIELDS_XYZRGB, points_3d)
    else:
        pointcloud_msg = pc2.create_cloud(header, _FIELDS_XYZRGB, points_3d)
    return pointcloud_msg


def rectify_image(image, info):
    # Initialisiere die rectifizierten Abbildungs-Maps für die Bildrectifizierung
    map_x, map_y = cv.initUndistortRectifyMap(info.cam_matrix, info.dist_coeff,
                                            info.rect_matrix, info.proj_matrix,
                                            (info.width, info.height), cv.CV_32FC1)
    # Wende die Abbildungs-Maps auf das Bild an, um es zu rectifizieren
    rectified_image = cv.remap(image, map_x, map_y, cv.INTER_LINEAR)
    return rectified_image

def scale_img(img, scale_percent=66.67):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

class StereoNode:
    def __init__(self, cam_name_1, cam_name_2, config, disparity_scaling, publish_pointcloud, publish_cam_1, publish_cam_2, cut_image):
        self.camera_1_name = cam_name_1
        self.camera_2_name = cam_name_2
        self.camera_1_info = None
        self.camera_1_image = None
        self.camera_2_info = None
        self.camera_2_image = None
        self.scaling_disparity=disparity_scaling
        # Subscriber für "camera_info" and "image_raw" Topics
        self.camera_1_info_subscriber = rospy.Subscriber(f'/camera/{self.camera_1_name}/camera_info',
                                                         CameraInfo, self.camera_info_callback)
        self.camera_2_info_subscriber = rospy.Subscriber(f'/camera/{self.camera_2_name}/camera_info',
                                                         CameraInfo, self.camera_info_callback)
        self.image_1_subscriber = message_filters.Subscriber(f'/camera/{self.camera_1_name}/image_raw', Image)
        self.image_2_subscriber = message_filters.Subscriber(f'/camera/{self.camera_2_name}/image_raw', Image)
        self.disp_image_pub = rospy.Publisher('/camera/stereo/disparity_image', Image, queue_size=10)

        self.stereo = cv2.StereoBM_create(numDisparities=config.numDisparities,
                                          blockSize=config.blockSize)
        self.config = config
        self.disp_depth_matrix = None
        self.roi1 = None
        self.roi2 = None
        # Desired results
        self.timestamp = None
        self.disparity = None
        self.points = None
        self.camera_1_image_rect = None
        self.camera_2_image_rect = None
        self.disp_img = None
        self.publish_pcl = publish_pointcloud
        if self.publish_pcl:
            self.disp_points_pub = rospy.Publisher('/camera/stereo/points', PointCloud2, queue_size=10)

        self.publish_cam_1 = publish_cam_1
        if self.publish_cam_1:
            self.cam_1_rect_image_pub = rospy.Publisher(f'/camera/{self.camera_1_name}/image_rect', Image,
                                                        queue_size=10)

        self.publish_cam_2 = publish_cam_2
        if self.publish_cam_2:
            self.cam_2_rect_image_pub = rospy.Publisher(f'/camera/{self.camera_2_name}/image_rect', Image,
                                                        queue_size=10)

        self.cut_image = cut_image
        self.waiting_for_data()
        self.setup_matcher()


    def setup_matcher(self):
        """ Strange notation to drop all results but disp_depth not of stereo_rectify """
        _, _, _, _, \
            self.disp_depth_matrix, \
            self.roi1, self.roi2 = \
            cv2.stereoRectify(cameraMatrix1=self.camera_1_info.cam_matrix,
                              distCoeffs1=self.camera_1_info.dist_coeff,
                              cameraMatrix2=self.camera_2_info.cam_matrix,
                              distCoeffs2=self.camera_2_info.dist_coeff,
                              imageSize=(self.camera_1_info.width, self.camera_1_info.height),
                              R=self.config.rotation,
                              T=self.config.translation,
                              alpha=1)
        self.stereo.setPreFilterType(self.config.preFilterType)
        self.stereo.setPreFilterSize(self.config.preFilterSize)
        self.stereo.setPreFilterCap(self.config.preFilterCap)
        self.stereo.setTextureThreshold(self.config.textureThreshold)
        self.stereo.setUniquenessRatio(self.config.uniquenessRatio)
        self.stereo.setSpeckleRange(self.config.speckleRange)
        self.stereo.setSpeckleWindowSize(self.config.speckleWindowSize)
        self.stereo.setDisp12MaxDiff(self.config.disp12MaxDiff)
        self.stereo.setMinDisparity(self.config.minDisparity)
        print("Matcher is configured, start matching ...")

    def read_cameras(self):
        ts = message_filters.ApproximateTimeSynchronizer([self.image_1_subscriber, self.image_2_subscriber], queue_size=10, slop=0.5)
        ts.registerCallback(self.image_callback)
        rospy.spin()

    def compute_images(self):
        # Warte auf Bilder von beiden Kamera-Topics
        if self.camera_1_image is not None and self.camera_2_image is not None:
            # Konvertiere die Bilder in Graustufen
            self.camera_1_image_rect = rectify_image(self.camera_1_image, self.camera_1_info)
            self.camera_2_image_rect = rectify_image(self.camera_2_image, self.camera_2_info)
            imgL_gray_scaled = scale_img(cv2.cvtColor(self.camera_1_image_rect, cv2.COLOR_BGR2GRAY),
                                         scale_percent=self.scaling_disparity)
            imgR_gray_scaled = scale_img(cv2.cvtColor(self.camera_2_image_rect, cv2.COLOR_BGR2GRAY),
                                         scale_percent=self.scaling_disparity)
            # Calculating disparity using the StereoBM algorithm
            # 40 Milliseconds
            start_time = datetime.datetime.now()
            self.disparity = self.stereo.compute(imgL_gray_scaled, imgR_gray_scaled).astype(np.float32)
            end_time = datetime.datetime.now()
            time_diff = end_time - start_time
            milliseconds = int(time_diff.total_seconds() * 1000)
            #print(f"Disparity: {milliseconds} Millisekunden")
            disparity_img = None
            disparity_img = cv2.normalize(self.disparity, disparity_img, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
            disparity_img = np.uint8(disparity_img)
            self.disp_img = cv2.applyColorMap(disparity_img, cv2.COLORMAP_INFERNO)
            self.publish_topics()

    def publish_topics(self):
        try:
            if self.publish_cam_1:
                if self.cut_image:
                    self.camera_1_image_rect = self.cut_rect_image(self.camera_1_image_rect, self.camera_1_info)
                left_rect_img_msg = bridge.cv2_to_imgmsg(self.camera_1_image_rect, encoding="passthrough")
                left_rect_img_msg.header.frame_id = self.camera_1_info.frame_id
                left_rect_img_msg.header.stamp = self.timestamp
                self.cam_1_rect_image_pub.publish(left_rect_img_msg)



            if self.publish_cam_2:
                if self.cut_image:
                    self.camera_2_image_rect = self.cut_rect_image(self.camera_2_image_rect, self.camera_2_info)
                left_rect_img_msg = bridge.cv2_to_imgmsg(self.camera_1_image_rect, encoding="passthrough")
                left_rect_img_msg.header.frame_id = self.camera_1_info.frame_id
                left_rect_img_msg.header.stamp = self.timestamp
                self.cam_2_rect_image_pub.publish(left_rect_img_msg)

            # Disparity Image
            disp_img_msg = bridge.cv2_to_imgmsg(self.disp_img, encoding='passthrough')
            self.disp_image_pub.publish(disp_img_msg)

            if self.publish_pcl:
            # Pointcloud - 12 Milliseconds
                pclmsg = PointCloud2()
                pclmsg.header.stamp = self.timestamp
                pclmsg.header.frame_id = "map"
                self.points = cv2.reprojectImageTo3D(self.disparity, self.disp_depth_matrix)
                pclmsg.width = self.points.shape[1]
                pclmsg.height = self.points.shape[0]
                pclmsg.fields = _FIELDS_XYZ
                pclmsg.is_bigendian = False
                pclmsg.is_dense = False
                pclmsg.point_step = 12
                pclmsg.row_step = 12 * self.points.shape[1]
                pclmsg.data = np.asarray(self.points, np.float32).tobytes()
                self.disp_points_pub.publish(pclmsg)

        except CvBridgeError as e:
            print(e)

    def cut_rect_image(self, image, info):
        """
        Schneide das Bild gemäß dem definierten ROI-Rechteck aus
        """
        x, y, w, h = info.roi.x_offset, info.roi.y_offset, \
           info.roi.width, info.roi.height
        image = image[y:y + h, x:x + w]
        return image

    def camera_info_callback(self, camera_info_msg):
        """
        Speichere die Kamerainformationen in der globalen Variable
        Args:
            camera_info_msg:
        """
        cam_name = camera_info_msg.header.frame_id
        if "left" in cam_name:
            self.camera_1_info = CamInfo(camera_info_msg)
        elif "right" in cam_name:
            self.camera_2_info = CamInfo(camera_info_msg)
        else:
            print("cam_name and left/right assignment doesn't match - info.")

    def image_callback(self, image_msg_left, image_msg_right):
        """
        Callback-Funktion für das "image_raw"-Topic
        Args:
            image_msg:
        """
        start_time = datetime.datetime.now()
        # Konvertiere das Bild von ROS-Nachricht in das OpenCV-Format
        self.camera_1_image = bridge.imgmsg_to_cv2(image_msg_left, desired_encoding="passthrough")
        self.camera_2_image = bridge.imgmsg_to_cv2(image_msg_right, desired_encoding="passthrough")
        self.timestamp = image_msg_left.header.stamp
        self.compute_images()
        end_time = datetime.datetime.now()
        time_diff = end_time - start_time
        milliseconds = int(time_diff.total_seconds() * 1000)
        print(f"Stereo: {milliseconds} Millisekunden")

    def waiting_for_data(self):
        while (self.camera_1_info is None or self.camera_2_info is None) and not rospy.is_shutdown():
            time.sleep(1)
            print("Wait for data")

if __name__ == '__main__':
    # Define workspace & open config
    package_path = rospkg.RosPack().get_path('disparity')

    config_path = f'{package_path}/config/stereo_algo_param.yaml'
    with open(config_path) as f:
        config = StereoConfig(yaml.safe_load(f))
    disp_scaling = rospy.get_param('/stereo/disparity_image/scaling')
    publish_pcl = rospy.get_param('/stereo/disparity_image/publish_pointcloud')
    publish_cam_1 = rospy.get_param('/stereo/disparity_image/publish_cam_1')
    publish_cam_2 = rospy.get_param('/stereo/disparity_image/publish_cam_2')
    cut_image = rospy.get_param('/stereo/disparity_image/cut_image')
    camera_name_1 = "stereo_left"
    camera_name_2 = "stereo_right"
    bridge = CvBridge()
    rospy.init_node(f'stereo_disparity')
    stereo_topics = StereoNode(cam_name_1=camera_name_1,
                               cam_name_2=camera_name_2,
                               config=config,
                               disparity_scaling=disp_scaling,
                               publish_pointcloud=publish_pcl,
                               publish_cam_1=publish_cam_1,
                               publish_cam_2=publish_cam_2,
                               cut_image=cut_image)
    stereo_topics.read_cameras()
