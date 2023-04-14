from hpe3d.model import HumanDetector, HumanPoseDetector2D, Triangulator
from hpe3d.utils import camera_utils
from hpe3d.logger import Logger
import yaml
import numpy as np

from typing import List
import numpy.typing as npt


class HumanPoseDetector3D():
    """
    HumanPoseDetector3D is made of three parts:
    1. Human detector: Detect humans in the image and crop the region of interest.
    2. Human pose detector 2D: Detect the 2D human pose in the cropped image.
    3. Triangulator: Triangulate the 2D human poses from multiple views to 3D human pose.
    """
    
    def __init__(self, human_detector: HumanDetector, human_pose_detector_2d: HumanPoseDetector2D, triangulator: Triangulator, device_name: str = 'cpu') -> None:
        """
        Args:
            human_detector: Human detector object.
            human_pose_detector_2d: Human pose detector 2D object.
            triangulator: Triangulator object.
            device_name: Device name. It can be either 'cpu' or 'cuda'.
        
        Returns:
            None
        """
        
        self.logger = Logger(__name__)
        self.human_detector = human_detector
        self.human_pose_detector_2d = human_pose_detector_2d
        self.triangulator = triangulator
    
    def set_human_detector_logging_level(self, level: int) -> None:
        """
        Set the logging level for the human detector.

        Args:
            level: Logging level.

        Returns:
            None
        """
        
        self.human_detector.set_logging_level(level)

    def set_human_pose_detector_2d_logging_level(self, level: int) -> None:
        """
        Set the logging level for the human pose detector 2D.

        Args:
            level: Logging level.

        Returns:
            None
        """
        
        self.human_pose_detector_2d.set_logging_level(level)

    def set_triangulator_logging_level(self, level: int) -> None:
        """
        Set the logging level for the triangulator.
        """

        self.triangulator.set_logging_level(level)

    def eval(self, multiview: List[npt.NDArray[np.uint8]]) -> npt.NDArray[np.single]:
        """
        Evaluate the 3D human pose from multiple views.

        Args:
            multiview: List of images from multiple views.

        Returns:
            human_pose_3d: 3D human pose.
        """
        
        multiview_poses_2d = []
        for camera_idx, singleview in enumerate(multiview):
            human_detections = self.human_detector.eval(singleview)
            if len(human_detections) == 0:
                self.logger.warn(f'No human detected in camera {camera_idx}')
                self.logger.warn(f'Camera {camera_idx} skipped')
                continue
            
            human_poses_2d = self.human_pose_detector_2d.eval(singleview, human_detections)
            multiview_poses_2d.append(human_poses_2d)

        human_pose_3d = self.triangulator.eval(multiview_poses_2d)
        return human_pose_3d
    
class HumanPoseDetector3DBuilder():
    """
    Builds a HumanPoseDetector3D object.
    
    HumanPoseDetector3DBuilder is made of three parts:
    1. Human detector: Detect humans in the image and crop the region of interest.
    2. Human pose detector 2D: Detect the 2D human pose in the cropped image.
    3. Triangulator: Triangulate the 2D human poses from multiple views to 3D human pose.
    """

    def __init__(self) -> None:
        self.logger = Logger(__name__)
        self.human_detection_model_path = None
        self.human_pose_detection_2d_model_path = None
        self.camera_params = None
        self.device_name = 'cpu'

    def load_human_detection_model(self, human_detection_model_path: str) -> None:
        """
        Load the human detection model path. 
        
        Args:
            human_detection_model_path (str): Path to the human detection model. It should be a directory containing
                the tensorrt engine file and pipline configurations. It can be generated using mmdeploy.

        Returns:
            None
        """
        
        self.human_detection_model_path = human_detection_model_path

    def load_human_pose_detection_2d_model(self, human_pose_detection_2d_model_path: str) -> None:
        """
        Load the human pose detection 2D model path.

        Args:
            human_pose_detection_2d_model_path (str): Path to the human pose detection 2D model. It should be a directory containing 
                the tensorrt engine file and pipline configurations. It can be generated using mmdeploy.

        Returns:
            None
        """

        self.human_pose_detection_2d_model_path = human_pose_detection_2d_model_path

    def load_camera_params(self, camera_params_path: str) -> None:
        """
        Load the camera parameters.

        Args:
            camera_params_path (str): Path to the camera parameters. It should be a yaml file containing the camera parameters.

        Returns:
            None
        """

        self.camera_params = camera_utils.get_camera_params(camera_params_path)

    def use_device(self, device_name: str) -> None:
        """
        Use the device for inference.

        Args:
            device_name (str): Device name. It can be either 'cpu' or 'cuda'.

        Returns:
            None
        """

        self.device_name = device_name

    def build(self) -> HumanPoseDetector3D:
        """
        Build the HumanPoseDetector3D object.

        Returns:
            HumanPoseDetector3D: HumanPoseDetector3D object.
        """
        
        if self.human_detection_model_path is None:
            raise ValueError('Human detector model path not set')
        if self.human_pose_detection_2d_model_path is None:
            raise ValueError('Human pose detector 2D model path not set')
        if self.camera_params is None:
            raise ValueError('Camera params not set')
        
        if self.device_name not in ['cpu', 'cuda']:
            raise ValueError('Device name not supported')
        
        if self.device_name == 'cpu':
            self.logger.warn('Running on CPU. This will be slow')
        
        human_detector = HumanDetector(self.human_detection_model_path, self.device_name)
        human_pose_detector_2d = HumanPoseDetector2D(self.human_pose_detection_2d_model_path, self.device_name)
        triangulator = Triangulator(self.camera_params)

        return HumanPoseDetector3D(human_detector, human_pose_detector_2d, triangulator, self.device_name)
    
    def build_from_config(self, config_file_path):
        pass
