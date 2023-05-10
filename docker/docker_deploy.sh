xhost local:root
docker run --gpus 'all,"capabilities=compute,utility,graphics"' \
  --network host \
  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY \
  -v ./data:/root/workspace/human_pose_estimation_3d/data \
  -it \
  --name cobot_deploy \
  cobot/hpe3d:latest 
