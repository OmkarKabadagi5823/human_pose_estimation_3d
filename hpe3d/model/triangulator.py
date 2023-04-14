import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from itertools import combinations
from hpe3d.logger import Logger

from typing import List, Dict
import numpy.typing as npt


class Triangulator():
    """
    Triangulates 3D poses from 2D poses.
    """

    def __init__(self, camera_params: Dict) -> None:
        """
        Args:
            camera_params: The camera parameters of the cameras used to capture the images. Should contain both intrinsics and extrinsics.

        Returns:
            None
        """

        self.logger = Logger(__name__)
        self.camera_params = camera_params

    def __dimensionality_filter(self, camera_idx1: int, camera_idx2: int, multiview_poses_2d: List[List[npt.NDArray[np.single]]]) -> bool:
        """
        Checks if the number of keypoints in the 2D poses of the two cameras are the same. (Internal Function)

        Args:
            camera_idx1: The index of the first camera.
            camera_idx2: The index of the second camera.
            multiview_poses_2d: The 2D poses of humans from all the camera views. [camera_idx][human_idx][keypoint_idx][py/px]

        Returns:
            A boolean value indicating whether the two cameras have the same number of keypoints.
        """
        
        if multiview_poses_2d[camera_idx1][0].shape != multiview_poses_2d[camera_idx2][0].shape:
            self.logger.warning(f'Camera {camera_idx1} and {camera_idx2} have different number of keypoints. This is not expected.')
            return False
        
        return True
    
    def __reprojection_filter(self, camera_idx1: int, camera_idx2: int, multiview_poses_2d: List[List[npt.NDArray[np.single]]]) -> bool:
        """
        Checks if 2D reprojection of the estimated 3D pose of the two cameras are within threshold of 
        previously detected 2D keypoints. (Internal Function) 

        Args:
            camera_idx1: The index of the first camera.
            camera_idx2: The index of the second camera.
            multiview_poses_2d: The 2D poses of humans from all the camera views. [camera_idx][human_idx][keypoint_idx][py/px]

        Returns:
            A boolean value indicating whether the reprojection error is within threshold.
        """
        
        camera1_params = self.camera_params[camera_idx1]
        camera2_params = self.camera_params[camera_idx2]

        r1_vec = Rotation.from_matrix(camera1_params["R"]).as_rotvec()
        t1_vec = np.array(camera1_params["t"]) / 100.0
        r2_vec = Rotation.from_matrix(camera2_params["R"]).as_rotvec()
        t2_vec = np.array(camera2_params["t"]) / 100.0

        pose_3d = self.__triangulate_pair(camera_idx1, camera_idx2, multiview_poses_2d)
        
        reprojected_2d_camera1 = np.squeeze(
            cv2.projectPoints(
                pose_3d, 
                r1_vec, t1_vec, 
                np.array(camera1_params['K']), np.array(camera1_params['distCoef'])
            )[0]
        )
        reprojected_2d_camera2 = np.squeeze(
            cv2.projectPoints(
                pose_3d,
                r2_vec, t2_vec,
                np.array(camera2_params['K']), np.array(camera2_params['distCoef'])
            )[0]
        )
        
        error_camera1 = np.linalg.norm(reprojected_2d_camera1 - multiview_poses_2d[camera_idx1][0])
        error_camera2 = np.linalg.norm(reprojected_2d_camera2 - multiview_poses_2d[camera_idx2][0])

        self.logger.debug(f'Error camera 1: {np.round(float(error_camera1), decimals=4)}, Error camera 2: {np.round(float(error_camera2), decimals=4)}')

        if error_camera1 > 100 or error_camera2 > 100:
            return False
    
        return True
    
    def __pair_filter(self, camera_idx1: int, camera_idx2: int, multiview_poses_2d: List[List[npt.NDArray[np.single]]]) -> bool:
        """
        Checks if the two cameras are suitable for triangulation. (Internal Function)

        Args:
            camera_idx1: The index of the first camera.
            camera_idx2: The index of the second camera.
            multiview_poses_2d: The 2D poses of humans from all the camera views. [camera_idx][human_idx][keypoint_idx][py/px]

        Returns:
            A boolean value indicating whether the two cameras are suitable for triangulation.
        """
        
        filters = [self.__dimensionality_filter, self.__reprojection_filter]
        
        for f in filters:
            if not f(camera_idx1, camera_idx2, multiview_poses_2d):
                self.logger.debug(f'Filter {f.__name__} failed on pair {camera_idx1}, {camera_idx2}')
                return False
            
        return True

    def __triangulate_pair(self, camera_idx1: int, camera_idx2: int, multiview_poses_2d: List[List[npt.NDArray[np.single]]]) -> npt.NDArray[np.single]:
        """
        Triangulates a 3D pose from 2D poses from two views. (Internal Function)

        Args:
            camera_idx1: The index of the first camera.
            camera_idx2: The index of the second camera.
            multiview_poses_2d: The 2D poses of humans from all the camera views. [camera_idx][human_idx][keypoint_idx][py/px]

        Returns:
            A 3D pose. [keypoint_idx][x/y/z]
        """
        
        P1 = self.camera_params[camera_idx1]["P"]
        P2 = self.camera_params[camera_idx2]["P"]

        x1 = multiview_poses_2d[camera_idx1][0][:, 0:2]
        x2 = multiview_poses_2d[camera_idx2][0][:, 0:2]

        joints3dest = cv2.triangulatePoints(P1, P2, x1.T, x2.T)
        joints3dest /= joints3dest[3]

        return joints3dest[0:3].T

    def set_logging_level(self, level: int) -> None:
        """
        Sets the logging level of the triangulation module.

        Args:
            level: The logging level.

        Returns:
            None
        """
        self.logger.setLevel(level)

    def eval(self, multiview_poses_2d: List[List[npt.NDArray[np.single]]]):
        """
        Estimates a 3D pose from 2D poses from multiple views.

        Args:
            multiview_poses_2d: The 2D poses of humans from all the camera views. [camera_idx][human_idx][keypoint_idx][py/px]

        Returns:
            A 3D pose. [keypoint_idx][x/y/z]
        """
        
        s = np.zeros((17, 3))
        num_views = len(multiview_poses_2d)
        
        view_combinations = combinations(range(num_views), 2)
        filtered_combinations = filter(lambda x: self.__pair_filter(x[0], x[1], multiview_poses_2d), view_combinations)
        
        num_processed_combinations = 0
        for camera_idx1, camera_idx2 in filtered_combinations:
            s += self.__triangulate_pair(camera_idx1, camera_idx2, multiview_poses_2d)
            num_processed_combinations += 1

        self.logger.debug(f'Number of processed combinations: {num_processed_combinations}')

        if num_processed_combinations == 0:
            self.logger.warning('No valid triangulation found. Returning zeros.')
            return s
        
        return s / num_processed_combinations