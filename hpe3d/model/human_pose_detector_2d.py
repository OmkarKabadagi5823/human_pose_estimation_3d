import cv2
import numpy as np
from mmdeploy_python import PoseDetector
from hpe3d.logger import Logger

from typing import List
import numpy.typing as npt

class HumanPoseDetector2D():
    """
    A wrapper class for mmdeploy's PoseDetector model to detect 2D human keypoints.
    """
    
    def __init__(self, model_path: str = None, device_name: str = 'cpu') -> None:
        """
        Args:
            model_path: Path to the mmdeploy model directory.
            device_name: Name of the device to run the model on. Can be 'cpu' or 'cuda'.

        Returns:
            None
        """

        self.logger = Logger(__name__)
        self.detector = PoseDetector(model_path=model_path, device_name=device_name, device_id=0)

    def set_logging_level(self, level: int) -> None:
        """
        Sets the logging level of the logger.

        Args:
            level: The logging level to set.

        Returns:
            None
        """
        
        self.logger.setLevel(level)

    def eval(self, img: np.ndarray, human_bboxes: List[npt.NDArray[np.single]]) -> List[npt.NDArray[np.single]]:
        """
        Detects 2D human keypoints in a given image.

        Args:
            img: The image to detect humans in.
            human_bboxes: The bounding boxes of the humans to detect keypoints for. List of [left, top, right, bottom, score].

        Returns:
            A list of 2D keypoints for each human. Each set of keypoints is an array of shape (num_keypoints, 2).
        """

        results = []
        for human_bbox in human_bboxes:
            result = self.detector(img, human_bbox[0:4])
            _, point_num, _ = result.shape
            points = result[:, :, :2].reshape(point_num, 2)
            results.append(points)

        self.logger.debug(f'Found {len(results)} humans:\n{results}')
            
        return results

    @staticmethod
    def draw_keypoints(img, multi_human_keypoints: List[npt.NDArray[np.single]]) -> None:
        """
        Draws the keypoints on the image. (Static Method)

        Args:
            img: The image to draw the keypoints on.
            multi_human_keypoints: A list of 2D keypoints for each human. Each set of keypoints is an array of shape (num_keypoints, 2).

        Returns:
            None
        """
        
        for single_human_keypoints in multi_human_keypoints:
            for [x, y] in single_human_keypoints.astype(int):
                cv2.circle(img, (x, y), 5, (0, 255, 0), 4)