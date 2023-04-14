import cv2
import numpy as np
from mmdeploy_python import Detector
from hpe3d.logger import Logger

from typing import List, Tuple
import numpy.typing as npt

class HumanDetector():
    """
    A wrapper class for the mmdeploy's Detector model that is used to detect humans in a given image.

    Methods:
        eval: Detects humans in a given image.
        set_logging_level: Sets the logging level of the logger.
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
        self.detector = Detector(model_path=model_path, device_name=device_name, device_id=0)

    def __filter(self, bboxes_with_labels: List[Tuple[npt.NDArray[np.single], int]]) -> bool:
        """
        Filters the bounding boxes and labels returned by the detector model. (Internal Function)

        Args:
            bboxes_with_labels: A list of tuples containing the bounding boxes and labels.

        Returns:
            A boolean value indicating whether the bounding box should be kept or not.
        """

        bbox, label_id = bboxes_with_labels

        area = HumanDetector.area_bbox(bbox)

        if((label_id == 0) and (bbox[4] > 0.5) and (area > 10000)):
            return True
        else:
            return False
        
    def set_logging_level(self, level: int) -> None:
        """
        Sets the logging level of the logger.

        Args:
            level: The logging level to set.

        Returns:
            None
        """
        self.logger.setLevel(level)
        
    def eval(self, img: npt.NDArray[np.uint8]) -> List[npt.NDArray[np.single]]:
        """
        Detects humans in a given image.

        Args:
            img: The image to detect humans in.
        
        Returns:
            A list of bounding boxes. Each bounding box is a list of the form [left, top, right, bottom, score].
        """

        bboxes, labels, _ = self.detector(img)
        results = zip(bboxes, labels)
        results = filter(self.__filter, results)
        results = sorted(results, key=lambda x: HumanDetector.area_bbox(x[0]), reverse=True)
        results = list(map(lambda x: x[0], results))
        
        def __format(bbox: npt.NDArray[np.single]) -> List[npt.NDArray[np.single]]:
            """
            Formats the bounding box. (Internal Function)

            Args:
                bbox: The bounding box to format. [left, top, right, bottom, score]

            Returns:
                The formatted bounding box.
            """
            
            mask = np.array([True, True, True, True, False])
            bbox[mask] = np.round(bbox[mask], 0)
            return bbox
   
        formatted_results = list(map(__format, results))

        self.logger.debug(f'Found {len(formatted_results)} humans:\n{formatted_results}')

        return results

    @staticmethod
    def area_bbox(bbox: npt.ArrayLike) -> float:
        """
        Calculates the area of a bounding box. (Static Method)

        Args:
            bbox: The bounding box to calculate the area of. [left, top, right, bottom, score]
        
        Returns:
            The area of the bounding box.
        """

        return abs((bbox[2]-bbox[0]) * (bbox[3]-bbox[1]))
    
    @staticmethod
    def draw_bboxs(img: npt.NDArray[np.uint8], bboxes: List[npt.NDArray[np.single]]) -> None:
        """
        Draws the bounding boxes on the image. (Static Method)

        Args:
            img: The image to draw the bounding boxes on.
            bboxes: The bounding boxes to draw. List of [left, top, right, bottom, score].

        Returns:
            None
        """
        
        for bbox in bboxes:
            [left, top, right, bottom]= bbox[0:4].astype(int)
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)