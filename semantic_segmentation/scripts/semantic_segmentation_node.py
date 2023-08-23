#!/usr/bin/env python3

import sys
import numpy as np
import torch
import torch.nn.functional as F
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image
import pidnet
import cv2 as cv
import rospkg
import yaml

def image_callback(image_msg):
    # Konvertiere das Bild von ROS-Nachricht in das OpenCV-Format
    image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")

    height, width = image.shape[0], image.shape[1]
    new_height, new_width = height, width
    while new_height % 8 != 0:
        new_height -= 1
    while new_width % 8 != 0:
        new_width -= 1
    x_offset = (width - new_width) // 2
    y_offset = (height - new_height) // 2

    cropped_image = image[y_offset:y_offset + new_height, x_offset:x_offset + new_width]



    with torch.no_grad():
        sv_img = np.zeros_like(cropped_image).astype(np.uint8)
        img = input_transform(cropped_image)
        img = img.transpose((2, 0, 1)).copy()
        img = torch.from_numpy(img).unsqueeze(0).cuda()
        pred = model(img)
        pred = F.interpolate(pred, size=img.size()[-2:],
                             mode='bilinear', align_corners=True)
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

        for i, color in enumerate(color_map_list):
            for j in range(3):
                sv_img[:, :, j][pred == i] = color_map_list[i][j]
        sem_seg_image = cv.cvtColor(sv_img, cv.COLOR_RGB2BGR)

    sem_seg_image = bridge.cv2_to_imgmsg(sem_seg_image, encoding="bgr8")
    sem_seg_image.header.frame_id = image_msg.header.frame_id
    sem_seg_image.header.stamp = image_msg.header.stamp
    sem_seg_publisher.publish(sem_seg_image)


def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if
                       (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    return model

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

def init_variables():
    camera_name = sys.argv[1]
    #camera_name = 'stereo_left'

    package_path = rospkg.RosPack().get_path('semantic_segmentation')
    #package_path = f"/home/ameise/setup/catkin_ws/src/semantic_segmentation"

    model = rospy.get_param(f'/{camera_name}/semantic_segmentation/model')
    #model = "s"
    if "m" in model:
        model_path = f"{package_path}/models/PIDNet_M_Cityscapes_val.pt"
    elif "l" in model:
        model_path = f"{package_path}/models/PIDNet_L_Cityscapes_val.pt"
    else:
        model_path = f"{package_path}/models/PIDNet_S_Cityscapes_val.pt"
    model = pidnet.get_pred_model(model, 19)
    model = load_pretrained(model, model_path).cuda()
    model.eval()

    mean = [0.485, 0.456, 0.406]

    std = [0.229, 0.224, 0.225]

    color_map_path = f"{package_path}/config/class_color_map.yaml"

    with open(color_map_path, 'r') as file:
        color_map = yaml.safe_load(file)

    color_map_list = []
    for key, value in color_map.items():
        color_map_list.append(tuple(value))

    bridge = CvBridge()

    return camera_name, model, mean, std, color_map_list, model, bridge

if __name__ == '__main__':

    camera_name, model, mean, std, color_map_list, model, bridge = init_variables()

    rospy.init_node(f'{camera_name}_sem_seg_node')

    # Erstelle Subscriber für "image_raw" Topics
    image_subscriber = rospy.Subscriber(f'/camera/{camera_name}/image_rect', Image, image_callback)

    # Erstelle Publisher für das Klassifizierte Bild
    sem_seg_publisher = rospy.Publisher(f'/camera/{camera_name}/semantic_segmentation', Image, queue_size=10)

    # Schleife für die ROS-Spin-Funktion
    while not rospy.is_shutdown():
        rospy.spin()