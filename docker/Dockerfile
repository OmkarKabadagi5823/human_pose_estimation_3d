FROM nvcr.io/nvidia/tensorrt:22.04-py3

ARG CUDA=11.3
ARG PYTHON_VERSION=3.8
ARG TORCH_VERSION=1.12.0
ARG TORCHVISION_VERSION=0.13.0
ARG ONNXRUNTIME_VERSION=1.8.1
ARG PPLCV_VERSION=0.7.0
ENV FORCE_CUDA="1"
ARG MMCV_VERSION=">=2.0.0rc2"
ARG MMENGINE_VERSION=">=0.3.0"

ENV DEBIAN_FRONTEND=noninteractive

### change the system source for installing libs
ARG USE_SRC_INSIDE=false
RUN if [ ${USE_SRC_INSIDE} == true ] ; \
    then \
        sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list ; \
        sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list ; \
        echo "Use aliyun source for installing libs" ; \
    else \
        echo "Keep the download source unchanged" ; \
    fi

### update apt and install libs
RUN apt-get update &&\
    apt-get install -y vim libsm6 libxext6 libxrender-dev libgl1-mesa-glx git wget libssl-dev libopencv-dev libspdlog-dev --no-install-recommends &&\
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build pyyaml numpy ipython cython typing typing_extensions mkl mkl-include ninja && \
    /opt/conda/bin/conda clean -ya

### change the pip source for installing packages
RUN if [ ${USE_SRC_INSIDE} == true ] ; \
    then \
        /opt/conda/bin/pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple; \
        echo "pip using tsinghua source" ; \
    else \
        echo "Keep pip the download source unchanged" ; \
    fi

### install pytorch openmim
RUN /opt/conda/bin/conda install pytorch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} cudatoolkit=${CUDA} -c pytorch -y \
    && /opt/conda/bin/pip install --no-cache-dir openmim

### pytorch mmcv onnxruntime
RUN /opt/conda/bin/mim install --no-cache-dir "mmcv"${MMCV_VERSION} onnxruntime-gpu==${ONNXRUNTIME_VERSION} mmengine${MMENGINE_VERSION}

ENV PATH /opt/conda/bin:$PATH
WORKDIR /root/workspace

### get onnxruntime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && tar -zxvf onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz

### cp trt from pip to conda
RUN cp -r /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorrt* /opt/conda/lib/python${PYTHON_VERSION}/site-packages/

### install mmdeploy
ENV ONNXRUNTIME_DIR=/root/workspace/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}
ENV TENSORRT_DIR=/workspace/tensorrt
ARG VERSION
RUN git clone -b main https://github.com/open-mmlab/mmdeploy &&\
    cd mmdeploy &&\
    if [ -z ${VERSION} ] ; then echo "No MMDeploy version passed in, building on main" ; else git checkout tags/v${VERSION} -b tag_v${VERSION} ; fi &&\
    git submodule update --init --recursive &&\
    mkdir -p build &&\
    cd build &&\
    cmake -DMMDEPLOY_TARGET_BACKENDS="ort;trt" .. &&\
    make -j$(nproc) &&\
    cd .. &&\
    /opt/conda/bin/mim install -e .

### build sdk
RUN git clone https://github.com/openppl-public/ppl.cv.git &&\
    cd ppl.cv &&\
    git checkout tags/v${PPLCV_VERSION} -b v${PPLCV_VERSION} &&\
    ./build.sh cuda

ENV BACKUP_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real/:$LD_LIBRARY_PATH

RUN cd /root/workspace/mmdeploy &&\
    rm -rf build/CM* build/cmake-install.cmake build/Makefile build/csrc &&\
    mkdir -p build && cd build &&\
    cmake .. \
        -DMMDEPLOY_BUILD_SDK=ON \
        -DMMDEPLOY_BUILD_EXAMPLES=ON \
        -DCMAKE_CXX_COMPILER=g++ \
        -Dpplcv_DIR=/root/workspace/ppl.cv/cuda-build/install/lib/cmake/ppl \
        -DTENSORRT_DIR=${TENSORRT_DIR} \
        -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
        -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
        -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
        -DMMDEPLOY_TARGET_BACKENDS="ort;trt" \
        -DMMDEPLOY_CODEBASES=all &&\
    make -j$(nproc) && make install &&\
    export SPDLOG_LEVEL=warn &&\
    if [ -z ${VERSION} ] ; then echo "Built MMDeploy for GPU devices successfully!" ; else echo "Built MMDeploy version v${VERSION} for GPU devices successfully!" ; fi

### install mmdetection and mmpose
RUN git clone -b 3.x https://github.com/open-mmlab/mmdetection.git && \
    git clone -b 1.x https://github.com/open-mmlab/mmpose.git && \
    mim install -v -e mmdetection/ && \
    mim install -v -e mmpose/

### ros installation
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list && \
    apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

RUN apt-get update && apt-get install -y --no-install-recommends ros-noetic-perception=1.5.0-1* ros-noetic-rviz* && \
    rm -rf /var/lib/apt/lists/* && \
    pip install rospkg

### download the model checkpoints
RUN wget https://download.openmmlab.com/mmdetection/v2.0/ssd/ssdlite_mobilenetv2_scratch_600e_coco/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth -P checkpoints && \
    wget https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_mobilenetv2_8xb64-210e_coco-256x192-55a04c35_20221016.pth -P checkpoints

ENV PYTHONPATH=/root/workspace/mmdeploy/build/lib
ENV LD_LIBRARY_PATH="/root/workspace/mmdeploy/build/lib:${BACKUP_LD_LIBRARY_PATH}"

COPY docker/entrypoint.sh /root/workspace/
COPY docker/model_convert_trt.sh /root/workspace/

### install human_pose_estimation_3d
ADD hpe3d /root/workspace/human_pose_estimation_3d/hpe3d
ADD config /root/workspace/human_pose_estimation_3d/config
ADD examples /root/workspace/human_pose_estimation_3d/examples
COPY pyproject.toml /root/workspace/human_pose_estimation_3d/

RUN pip install -e /root/workspace/human_pose_estimation_3d/

CMD ["/root/workspace/entrypoint.sh", "bash"]
