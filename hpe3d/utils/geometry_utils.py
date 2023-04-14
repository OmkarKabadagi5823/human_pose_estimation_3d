import numpy as np
from geometry_msgs.msg import Quaternion

from typing import Tuple

def quaternion_from_axis_angle(axis: Tuple[float, float, float], angle: float) -> Quaternion:
    """
    Calculates the quaternion from the axis and angle.

    Args:
        axis: The axis of rotation. [x, y, z]
        angle: The angle of rotation in radians.

    Returns:
        The quaternion as geometry_msgs/Quaternion.
    """
    
    xx, yy, zz = axis
    qx = xx * np.sin(angle / 2)
    qy = yy * np.sin(angle / 2)
    qz = zz * np.sin(angle / 2)
    qw = np.cos(angle / 2)
    q = np.array([qw, qx, qy, qz])
    q = q / np.linalg.norm(q)
    
    return Quaternion(q[1], q[2], q[3], q[0])
