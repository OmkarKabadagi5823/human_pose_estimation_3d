import rospy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point, Vector3, Quaternion
from ..utils.geometry_utils import *

import numpy.typing as npt

class RvizVisualizer():
    """
    Visualizes the human pose in rviz.
    """
    
    def __init__(self, pub_topic: str) -> None:
        """
        Args:
            pub_topic: The topic to publish the markers to.
        """
        
        rospy.init_node('human_rviz_visualizer', anonymous=True)

        self.markers_publisher = rospy.Publisher(pub_topic, MarkerArray, queue_size=10)
        self.marker_array = MarkerArray()
        self.marker_array.markers = []

    def update(self, human_pose_3d: npt.NDArray[np.single]) -> None:
        """
        Updates the human pose in rviz.

        Args:
            pose3d: The human pose. [num_joints][x/y/z]

        Returns:
            None
        """
        
        self.__create_human_skeleton(human_pose_3d)
        self.markers_publisher.publish(self.marker_array)

    def __create_sphere_list(self, id: int, points: npt.NDArray[np.single]) -> Marker:
        """
        Creates a sphere list marker representing the human joints. (Internal function)

        Args:
            id: The id of the marker.
            points: The points to visualize. [num_points][x/y/z]
        
        Returns:
            The marker.
        """
        
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = rospy.Time.now()
        marker.ns = 'human'
        marker.id = id
        marker.type = Marker.SPHERE_LIST
        marker.action = 0

        marker.points = list(map(lambda x: Point(x[0], x[1], x[2]), points))
        marker.pose.position = Point(0.0, 0.0, 2.0)
        marker.pose.orientation = quaternion_from_axis_angle([0.0, 1.0, 0.0], np.pi / 7.0)
        
        marker.scale = Vector3(0.05, 0.05, 0.05)
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)

        return marker
    
    def __create_line(self, id: int, point1: npt.ArrayLike, point2: npt.ArrayLike):
        """
        Creates a line marker representing joint links. (Internal function)

        Args:
            id: The id of the marker.
            point1: The first point of the line. [x/y/z]
            point2: The second point of the line. [x/y/z]

        Returns:
            The marker.
        """
        
        p1 = Point(point1[0], point1[1], point1[2])
        p2 = Point(point2[0], point2[1], point2[2])

        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = rospy.Time.now()
        marker.ns = 'human_skeleton'
        marker.id = id
        marker.type = Marker.LINE_LIST
        marker.action = 0

        marker.points = [p1, p2]
        marker.pose.position = Point(0.0, 0.0, 2.0)
        marker.pose.orientation = quaternion_from_axis_angle([0.0, 1.0, 0.0], np.pi / 7.0)
        marker.scale.x = 0.03
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)

        return marker
    
    def __create_human_skeleton(self, human_pose_3d: npt.NDArray[np.single]):
        """
        Creates the human skeleton markers. (Internal function)

        Args:
            pose3d: The human pose. [num_joints][x/y/z]

        Returns:
            None
        """
        
        self.marker_array.markers.append(self.__create_sphere_list(0, human_pose_3d))
        self.marker_array.markers.append(self.__create_line(1, human_pose_3d[1], human_pose_3d[0]))
        self.marker_array.markers.append(self.__create_line(2, human_pose_3d[2], human_pose_3d[1]))
        self.marker_array.markers.append(self.__create_line(3, human_pose_3d[3], human_pose_3d[2]))
        self.marker_array.markers.append(self.__create_line(4, human_pose_3d[4], human_pose_3d[0]))
        self.marker_array.markers.append(self.__create_line(5, human_pose_3d[5], human_pose_3d[4]))
        self.marker_array.markers.append(self.__create_line(6, human_pose_3d[6], human_pose_3d[5]))
        self.marker_array.markers.append(self.__create_line(7, human_pose_3d[7], human_pose_3d[8]))
        self.marker_array.markers.append(self.__create_line(8, human_pose_3d[9], human_pose_3d[8]))
        self.marker_array.markers.append(self.__create_line(9, human_pose_3d[10], human_pose_3d[9]))
        self.marker_array.markers.append(self.__create_line(10, human_pose_3d[11], human_pose_3d[10]))
        self.marker_array.markers.append(self.__create_line(11, human_pose_3d[12], human_pose_3d[8]))
        self.marker_array.markers.append(self.__create_line(12, human_pose_3d[13], human_pose_3d[12]))
        self.marker_array.markers.append(self.__create_line(13, human_pose_3d[14], human_pose_3d[13]))
        self.marker_array.markers.append(self.__create_line(14, human_pose_3d[8], human_pose_3d[0]))