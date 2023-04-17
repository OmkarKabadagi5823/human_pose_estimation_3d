python /root/workspace/mmdeploy/tools/deploy.py \
    /root/workspace/mmdeploy/configs/mmdet/detection/detection_tensorrt-fp16_static-320x320.py \
    /root/workspace/mmdetection/configs/ssd/ssdlite_mobilenetv2-scratch_8xb24-600e_coco.py \
    /root/workspace/checkpoints/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth \
    /root/workspace/mmdeploy/demo/resources/det.jpg \
    --work-dir /root/workspace/human_pose_estimation_3d/data/mmdeploy_models/mobilenetv2_det-fp16-static \
    --device cuda \
    --dump-info

python /root/workspace/mmdeploy/tools/deploy.py \
    /root/workspace/mmdeploy/configs/mmpose/pose-detection_tensorrt-fp16_static-256x192.py\
    /root/workspace/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_mobilenetv2_8xb64-210e_coco-256x192.py \
    /root/workspace/checkpoints/td-hm_mobilenetv2_8xb64-210e_coco-256x192-55a04c35_20221016.pth \
    /root/workspace/mmdeploy/demo/resources/human-pose.jpg \
    --work-dir /root/workspace/human_pose_estimation_3d/data/mmdeploy_models/mobilenetv2_pose-fp16-static \
    --device cuda \
    --dump-info
