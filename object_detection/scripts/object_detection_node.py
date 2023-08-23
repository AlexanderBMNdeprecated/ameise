#!/usr/bin/env python3

import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2D
from vision_msgs.msg import ObjectHypothesisWithPose
from vision_msgs.msg import Detection2DArray
import cv2 as cv
import random
from ultralytics import YOLO
import yaml
import rospkg

# Klasse zum Speichern der Kamerainformationen

# Callback-Funktion für das "image_raw"-Topic
def image_callback(image_msg):
    # Konvertiere das Bild von ROS-Nachricht in das OpenCV-Format
    image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")

    # Bild rectifizieren
    prediction = model.predict(source=image)

    detections_msg = Detection2DArray()
    detections_msg.header = image_msg.header

    for box_data in prediction[0].boxes:
        # Detection
        detection = Detection2D()

        # get label and score
        label = model.names[int(box_data.cls)]
        score = float(box_data.conf)

        # get boxes values
        box = box_data.xywh[0]
        detection.bbox.center.x = float(box[0])
        detection.bbox.center.y = float(box[1])
        detection.bbox.size_x = float(box[2])
        detection.bbox.size_y = float(box[3])

        # get track id
        #track_id = ""
        #if box_data.is_track:
        #    track_id = str(int(box_data.id))
        #detection.id = track_id

        # create hypothesis
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.id = label
        hypothesis.score = score
        detection.results.append(hypothesis)

        # append msg
        detections_msg.detections.append(detection)

        # draw boxes for debug
        if label not in class_to_color:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            box_data = random.randint(0, 255)
            class_to_color[label] = (r, g, box_data)
        color = class_to_color[label]

        min_pt = (round(detection.bbox.center.x - detection.bbox.size_x / 2.0),
                  round(detection.bbox.center.y - detection.bbox.size_y / 2.0))
        max_pt = (round(detection.bbox.center.x + detection.bbox.size_x / 2.0),
                  round(detection.bbox.center.y + detection.bbox.size_y / 2.0))
        cv.rectangle(image, min_pt, max_pt, color, 4)

        #label = "{} {:.2f}".format(label, score)
        pos = (min_pt[0], min_pt[1])
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(image, label, pos, font,
                    1, color, 2, cv.LINE_AA)


    # Veröffentliche das rectifizierte Bild auf einem neuen ROS-Topic
    object_publisher.publish(detections_msg)
    detection_image_msg = bridge.cv2_to_imgmsg(image, encoding="bgr8")
    detection_image_msg.header.frame_id = image_msg.header.frame_id
    detection_image_msg.header.stamp = image_msg.header.stamp
    detection_image_publisher.publish(detection_image_msg)

if __name__ == '__main__':
    # ROS-Initialisierung
    camera_name = sys.argv[1]
    model = rospy.get_param(f'/{camera_name}/object_detection/model')
    package_path = rospkg.RosPack().get_path('object_detection')


    #camera_name = 'stereo_left'
    #model = 's'
    #package_path = f"/home/ameise/setup/catkin_ws/src/object_detection/"

    if "n" in model:
        model_path = f"../models/yolov8n.pt"
    elif "s" in model:
        model_path = f"../models/yolov8s.pt"
    elif "m" in model:
        model_path = f"../models/yolov8m.pt"
    elif "l" in model:
        model_path = f"../models/yolov8l.pt"
    else:
        model_path = f"../models/yolov8x.pt"

    bridge = CvBridge()
    model = YOLO(model_path)



    # File path for loading the class_to_color dictionary
    colo_map_path = f"{package_path}/config/class_color_map.yaml"

    # Load the class_to_color dictionary from the YAML file
    with open(colo_map_path, "r") as f:
        class_to_color = yaml.load(f, Loader=yaml.SafeLoader)

    rospy.init_node(f'{camera_name}_object_detection_node')

    # Initialisiere die Variable für die Kamerainformationen
    cam_info = None

    # Erstelle Subscriber für "image_raw" Topics
    image_subscriber = rospy.Subscriber(f'/camera/{camera_name}/image_rect', Image, image_callback)

    # Erstelle Publisher für das Klassifizierte Bild
    object_publisher = rospy.Publisher(f'/camera/{camera_name}/objects', Detection2DArray, queue_size=10)
    detection_image_publisher = rospy.Publisher(f'/camera/{camera_name}/detection_image', Image, queue_size=10)

    # Schleife für die ROS-Spin-Funktion
    while not rospy.is_shutdown():
        rospy.spin()