#!/usr/bin/env python
import rospy
import numpy as np
import json
import requests
from sensor_msgs.msg import NavSatFix
from basler_cam.scripts.old_toDelete import environmentModel_data

send_to_endpoint = True
endpoint_path = 'http://192.168.15.3:5001/input/new'

objectId = 1

objectClass = "Bus"
data_source_description = "Bus GPS Position"
source_id = 48
rotation = np.asarray([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 1.0]])
translation = np.asarray([0,0,0])
sprinterX = 1.993
sprinterY = 5.910
sprinterZ = 2.725
velocity = 0


def callback(data):
    #rospy.loginfo(rospy.get_caller_id() + "I heard %s", data)
    object_data = environmentModel_data.EnvironmentModelObject(
        environmentModel_data.Header(
            objectId,
            objectClass,
            float(str(data.header.stamp.secs) + "." + str(data.header.stamp.nsecs)),
            environmentModel_data.DataSource(
                data_source_description,
                source_id)
        ),
        environmentModel_data.DataOrigin(
            rotation.tolist(), #correct rotation
            environmentModel_data.Position(
                (data.latitude,
                 data.longitude))
        ),
        environmentModel_data.Position(
            (data.latitude,
             data.longitude)
        ),
        environmentModel_data.Dimension(
            sprinterX,
            sprinterY,
            sprinterZ
        ),
        velocity,
        rotation.tolist(),
        translation.tolist()
    )
    json_str = json.dumps(object_data, default=lambda o: o.__dict__, indent=4)
    json_dict = json.loads(json_str)
    if send_to_endpoint:
        r = requests.post(endpoint_path, json=json_dict)
    else:
        print(json_dict)


if __name__ == '__main__':
    rospy.init_node('gps_sub', anonymous=True)
    rospy.Subscriber("/gps/fix", NavSatFix, callback)
    rospy.spin()