from hpe3d.model import HumanPoseDetector3DBuilder
from hpe3d.utils import dataset_utils
from hpe3d.logger import Logger


import cv2
import numpy as np
import time
import signal
import logging
import os

def main(args):
    logger = Logger(__name__)

    ## Get camera directories    
    dataset_dir = os.path.join(project_root_path(), 'data', 'datasets', 'walk_session_1')
    cam_dirs = dataset_utils.get_camera_directories(dataset_dir)

    ## Build the 3D human pose detector
    model_builder = HumanPoseDetector3DBuilder()
    model_builder.load_human_detection_model(args.human_detection_model)
    model_builder.load_human_pose_detection_2d_model(args.human_pose_detection_2d_model)
    model_builder.load_camera_params(args.camera_params)
    model_builder.use_device(args.device_name)
    human_pose_detector_3d = model_builder.build()
    
    ## Uncomment the following lines to debug each section of the model seperately
    # human_pose_detector_3d.set_human_detector_logging_level(logging.DEBUG)
    # human_pose_detector_3d.set_human_pose_detector_2d_logging_level(logging.DEBUG)
    # human_pose_detector_3d.set_triangulator_logging_level(logging.DEBUG)

    if args.visualize:
        from hpe3d.visualizer import RvizVisualizer
        visualizer = RvizVisualizer('/human_markers')
    
    s = time.time()

    for t, multiview in enumerate(dataset_utils.grab_frames(cam_dirs)):
        logger.info(f'Processing frame {t}')

        ## Read images
        multiview = list(map(lambda x: cv2.imread(x), multiview))

        ## Evaluate the 3D human pose
        human_pose_3d = human_pose_detector_3d.eval(multiview)

        ## Transform the 3D human pose to the world frame
        human_pose_3d = dataset_utils.transform_camera_to_world(human_pose_3d)

        ## Transform the 3D human pose to the Panoptic frame
        human_pose_3d = dataset_utils.coco_to_panoptic(human_pose_3d)
        
        ## Visualize the 3D human pose
        if args.visualize:
            visualizer.update(human_pose_3d)
    
    logger.info(f'FPS: {(t+1) / (time.time() - s)}')

def parse_args():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--human_detection_model', type=str, required=True)
    parser.add_argument('--human_pose_detection_2d_model', type=str, required=True)
    parser.add_argument('--camera_params', type=str, required=True)
    parser.add_argument('--device_name', type=str, default='cpu')
    parser.add_argument('--visualize', action='store_true', default=False)
    
    return parser.parse_args()

def project_root_path():
    current_dir = os.getcwd()
    while current_dir != '/':
        if os.path.exists(os.path.join(current_dir, 'pyproject.toml')):
            return current_dir

        current_dir = os.path.dirname(current_dir)

def sigint_handler(sig, frame):
    exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, sigint_handler)
    np.set_printoptions(suppress=True)
    args = parse_args()

    main(args)
