## Description
`hpe3d` is a python package for Multiview 3D human pose estimation on the TensorRt runtime. The model is a multi-stage model that uses a [`top-down approach`](https://arxiv.org/abs/2202.02656) to build a 3D pose.
This allows the model to be modular and configurable to use different models for different stages. It uses TensorRT models deployed using [mmdeploy](https://mmdeploy.readthedocs.io/en/stable/get_started.html).

## Goal of the Project
Estimating the full 3D pose of an object from multiple views is a difficult problem that often requires adapting the model to the specific environment where it is used. This is because the model depends on the camera configuration, such as the type, number and position of the cameras, and the choice of the coordinate system. Therefore, whenever the model is applied to a new setting, it has to be trained again with new data. This raises two challenges: how to create a suitable dataset for training that covers different poses and views of the object, and how to run the model on low-cost GPU devices that have limited memory and processing power. The goal of this project is to develop a model that can be easily deployed in different scenarios without requiring retraining and that can run on cheaper GPUs without compromising accuracy or speed.

## Getting Started
### Installation
```bash
git clone https://github.com/OmkarKabadagi5823/human_pose_estimation_3d.git
cd human_pose_estimation_3d
python -m build
cd dist
pip install human_pose_estimation_3d-<version>-py3-none-any.whl
```

### Usage
```python
from hpe3d.model import HumanPoseDetector3DBuilder

# Create a builder object
model_builder = HumanPoseDetector3DBuilder()
model_builder.load_human_detector_model("path/to/detector/model")
model_builder.load_human_pose_detector_2d_model("path/to/pose/detector/model")
model_builder.load_camera_params("path/to/camera/parameters")
human_pose_detector_3d = model_builder.build()

# Detect 3D pose from 3 views
view1 = cv2.imread("path/to/view1")
view2 = cv2.imread("path/to/view2")
view3 = cv2.imread("path/to/view3")
pose = human_pose_detector_3d.eval(view1, view2, view3)
```

For further details, please refer the inline documentation.

### Run the example
Before running the example, you need to download and convert the models into TensorRT models using the [model converter tool](https://mmdeploy.readthedocs.io/en/stable/02-how-to-run/convert_model.html) from mmdeploy. Choose one model each from MMDetection and MMPose from the [supported model list](https://mmdeploy.readthedocs.io/en/stable/03-benchmark/supported_models.html). You can also convert custom models by writing your own custom conversion configurations. In our case, we use SSDLite and MobileNetV2 model respectively. Place the converted models under `{hpe3d_root_dir}/data/mmdeploy_models`. You will also need the dataset to run the model.

usage:
```bash
cd examples
python walk_session_1_example.py \
--human_detection_model <path/to/human/detection/model> \
--human_pose_detection_2d_model <path/to/human/pose/detection/2d/model> \
--camera_params <path/to/camera/parameters> \
--device_name cuda \
--visualize # optional (Requires ROS and Rviz)
```

example:
```bash
cd examples
python walk_session_1_example.py \
--human_detection_model ../data/mmdeploy_models/mobilenetv2_det-fp16-static \
--human_pose_detection_2d_model ../data/mmdeploy_models/mobilenetv2_pose-fp16-static \
--camera_params ../config/camera_parameters.json \
--device_name cuda
```

## Using Docker Image
The docker image requires nvidida-container-runtime to be installed on the host machine. Please refer to the [official repository](https://github.com/NVIDIA/nvidia-container-runtime#installation) for installation instructions.

### Build docker image
```bash
docker build -t cobot/hpe3d:latest -f docker/Dockerfile .
```

### Run the docker container
```bash
bash docker/docker_deploy.sh
```

## Output format
The output of the model is a `15 x 3` ndarray which represents the `x`, `y` and `z` position coordinates of the `15` joints. The joints are ordered as follows:
-  0: hip (hip centre)
-  1: r_hip (right hip)
-  2: r_knee (right knee)
-  3: r_foot (right foot)
-  4: l_hip (left hip)
-  5: l_knee (left knee)
-  6: l_foot (left foot)
-  7: nose
-  8: c_shoulder (shoulder centre)
-  9: r_shoulder (right shoulder)
- 10: r_elbow (right elbow)
- 11: r_wrist (right wrist)
- 12: l_shoulder (left shoulder)
- 13: l_elbow (left elbow)
- 14: l_wrist (left wrist)
