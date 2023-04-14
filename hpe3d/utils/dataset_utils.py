import os
import os.path as osp
import numpy as np

from typing import List, Generator
import numpy.typing as npt

# Links of the human body skeleton
keypoints_links = [
    [0,1],
    [0,2],
    [1,3],
    [2,4],
    [1,2],
    [3,5],
    [4,6],
    [5,6],
    [5,7],
    [6,8],
    [7,9],
    [8,10],
    [5,11],
    [6,12],
    [11,12],
    [14,12],
    [16,14],
    [13,11],
    [15,13]
]


def get_camera_directories(dataset_root_path: str) -> List[str]:
    """
    Returns a list of camera directories in the dataset root path.

    Args:
        dataset_root_path: The path to the dataset root directory.
    
    Returns:
        A list of camera directories.
    """
    
    cam_dirs = []
    for cam_dir in os.listdir(dataset_root_path):
        if osp.isdir(osp.join(dataset_root_path, cam_dir)):
            cam_dirs.append(osp.join(dataset_root_path, cam_dir))
    
    cam_dirs.sort()
    return cam_dirs

def grab_frames(cam_dirs: List[str], skip: int = 1) -> Generator[List[str], None, None]:
    """
    Returns a generator that yields a list of frames from each camera.

    Args:
        cam_dirs: A list of camera directories.
        skip: The number of frames to skip between each frame.

    Returns:
        A generator that yields a list of frames from each camera. 
        It contains paths to the frames which need to read manually
    """
    
    streams = []

    for cam_dir in cam_dirs:
        image_paths = os.listdir(cam_dir)
        image_paths.sort()
        image_paths = list(map(lambda x: osp.join(cam_dir, x), image_paths))
        streams.append(image_paths)

    num_streams = len(streams)
    num_frames = [len(stream) for stream in streams]
    min_num_frames = np.min(num_frames)

    for t in range(0, min_num_frames, skip):
        multiview = [streams[stream_idx][t] for stream_idx in range(num_streams)]
        yield multiview

def transform_camera_to_world(points: npt.NDArray[np.single]) -> npt.NDArray[np.single]:
    """
    Transforms points from camera coordinates to world coordinates.

    Args:
        points: The points to transform. [num_points][x/y/z]
    
    Returns:
        The transformed points. [num_points][x/y/z]
    """

    transform = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    points = (transform @ points.T).T

    return points

def coco_to_panoptic(coco_kpts: npt.NDArray[np.single]) -> npt.NDArray[np.single]:
    """
    Transforms keypoints from COCO format to Panoptic format.

    Args:
        coco_kpts: The keypoints to transform. [num_keypoints][x/y/z]

    Returns:
        The transformed keypoints. [num_keypoints][x/y/z]
    """
    
    panoptic_kpts = np.zeros((15, 3), dtype=np.float64)

    panoptic_kpts[0] = (coco_kpts[11] + coco_kpts[12]) /2 
    panoptic_kpts[1] = coco_kpts[12]
    panoptic_kpts[2] = coco_kpts[14]
    panoptic_kpts[3] = coco_kpts[16]
    panoptic_kpts[4] = coco_kpts[11]
    panoptic_kpts[5] = coco_kpts[13]
    panoptic_kpts[6] = coco_kpts[15]
    panoptic_kpts[7] = coco_kpts[0]
    panoptic_kpts[8] = (coco_kpts[5] + coco_kpts[6]) / 2
    panoptic_kpts[9] = coco_kpts[6]
    panoptic_kpts[10] = coco_kpts[8]
    panoptic_kpts[11] = coco_kpts[10]
    panoptic_kpts[12] = coco_kpts[5]
    panoptic_kpts[13] = coco_kpts[7]
    panoptic_kpts[14] = coco_kpts[9]

    return panoptic_kpts