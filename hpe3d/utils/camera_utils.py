import json
import numpy as np

from typing import Dict
import numpy.typing as npt

def get_camera_params(camera_params_path: str) -> Dict:
    """
    Returns the camera parameters from a json file.

    Args:
        camera_params_path: The path to the json file containing the camera parameters.

    Returns:
        A list of camera parameters. Each camera parameter is a dictionary containing the following keys:
            - K: The camera intrinsics.
            - R: The camera rotation.
            - t: The camera translation.
            - P: The camera projection matrix.
            - distCoef: The distortion coefficients.
            - name: The name of the camera.
            - type: The camera type
            - resolution: The resolution of the camera
    """

    with open(camera_params_path, 'r') as f:
        cameras = json.load(f)['cameras']
        for camera in cameras:
            K = np.array(camera['K'], dtype=np.float64) # camera intrinsics
            R = np.array(camera['R'], dtype=np.float64) # camera rotation
            t = np.array(camera['t'], dtype=np.float64) / 100 # camera translation (in metres)

            camera['P'] = K @ np.hstack((R, t))

        return cameras
    
def fundamental_from_projections(P1: npt.NDArray[np.single], P2: npt.NDArray[np.single]):
    """
    Computes the fundamental matrix from two camera projection matrices.

    Args:
        P1: The first camera 3x4 projection matrix.
        P2: The second camera 3x4 projection matrix.

    Returns:
        The 3x3 fundamental matrix.
    """
    
    X = np.array([
        np.vstack((P1[1], P1[2])),
        np.vstack((P1[2], P1[0])),
        np.vstack((P1[0], P1[1]))
    ])

    Y = np.array([
        np.vstack((P2[1], P2[2])),
        np.vstack((P2[2], P2[0])),
        np.vstack((P2[0], P2[1]))
    ])

    F = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            XY = np.vstack((X[j], Y[i]))
            F[i, j] = np.linalg.det(XY)

    return F
