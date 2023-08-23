import rosbag
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

def read_point_cloud_from_bag(bag_file, topic):
    bag = rosbag.Bag(bag_file)
    for topic, msg, t in bag.read_messages(topics=[topic]):
        if topic == topic:
            header = msg.header
            fields = msg.fields
            point_step = msg.point_step
            row_step = msg.row_step
            data = msg.data

            # Konvertierung der Daten in ein Numpy-Array
            points = pc2.read_points(msg, skip_nans=True)
            point_data = []
            for point in points:
                point_data.append(point)

            # Hier kannst du die PointCloud-Daten weiterverarbeiten
            # Beispiel: Anzeigen der Header-Informationen
            print("Header:")
            print(header)

            # Beispiel: Anzeigen der Feldinformationen
            print("PointFields:")
            for field in fields:
                print(field)

            # Beispiel: Anzeigen der PointStep
            print("PointStep:")
            print(point_step)

            # Beispiel: Anzeigen der Daten
            print("PointCloud-Daten:")
            for point in point_data:
                print(point)

    bag.close()

# Beispielaufruf
bag_file = '/home/ameise/Downloads/sample_moriyama_150324.bag'  # Passe den Dateinamen an
topic = '/points_raw'  # Passe das gewünschte Topic an
read_point_cloud_from_bag(bag_file, topic)