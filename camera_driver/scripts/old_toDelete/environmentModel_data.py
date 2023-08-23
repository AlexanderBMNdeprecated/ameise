import numpy as np


class EnvironmentModelObject:
    def __init__(self, header, data_origin, abs_position, bbox_size, velocity, rotation, translation):
        self.header = header
        self.dataOrigin = data_origin
        self.absolutPosition = abs_position
        self.objectMeasurement = bbox_size
        self.velocity = velocity
        self.rotation = rotation
        self.translation = translation


class Header:
    def __init__(self, object_id, object_class, timestamp, data_source):
        self.objectId = int(object_id)
        self.objectClass = object_class
        self.lastSeenTimestamp = timestamp
        # define a standard for the time
        self.normalize_time()
        self.dataSource = data_source

    def normalize_time(self):
        """ Normalized to UNIX
        :return:
        """
        pass


class DataSource:
    def __init__(self, data_owner, source_id):
        self.dataOwner = data_owner
        self.sourceId = source_id


class DataOrigin:
    def __init__(self, origin_rotation, origin_position):
        self.originRotation = origin_rotation
        self.originPosition = origin_position


class Position:
    def __init__(self, pos):
        # Latitude, Longitude
        self.longitude = pos[0]
        self.latitude = pos[1]


class Dimension:
    def __init__(self, height, width, length):
        self.height = float(height)
        self.width = float(width)
        self.length = float(length)
