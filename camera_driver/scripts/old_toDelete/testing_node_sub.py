#!/usr/bin/env python3

import geometry_msgs
import tf2_ros
import tf

import os
import rospy
from sensor_msgs.msg import Image


def main():
    os.chdir("/home/ameise/catkin_ws/src/pano_pkg/scripts/")
    rospy.init_node('cam_node_sub', anonymous=True)
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    static_transformStamped = geometry_msgs.msg.TransformStamped()
    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = "alex_isses"
    static_transformStamped.child_frame_id = "oh_yes"

    static_transformStamped.transform.translation.x = 0
    static_transformStamped.transform.translation.y = 0
    static_transformStamped.transform.translation.z = 0
    quat = tf.transformations.quaternion_from_euler(0, 0, 0)
    static_transformStamped.transform.rotation.x = quat[0]
    static_transformStamped.transform.rotation.y = quat[1]
    static_transformStamped.transform.rotation.z = quat[2]
    static_transformStamped.transform.rotation.w = quat[3]
    broadcaster.sendTransform(static_transformStamped)

    rospy.Subscriber("Alex_hat_rieeeeeesig/image", Image, queue_size=1)
    rospy.spin()

if __name__ == '__main__':
    main()
