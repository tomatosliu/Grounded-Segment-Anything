FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Arguments to build Docker Image using CUDA
ARG USE_CUDA=0
ARG TORCH_ARCH=

ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA "${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST "${TORCH_ARCH}"
ENV CUDA_HOME /usr/local/cuda-11.6/

RUN mkdir -p /home/appuser/Grounded-Segment-Anything
COPY . /home/appuser/Grounded-Segment-Anything/

RUN apt-get update && apt-get install --no-install-recommends wget ffmpeg=7:* \
    libsm6=2:* libxext6=2:* git=1:* nano=2.* \
    vim=2:* -y \
    && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

WORKDIR /home/appuser/Grounded-Segment-Anything
RUN python -m pip install --no-cache-dir -e segment_anything -i https://pypi.tuna.tsinghua.edu.cn/simple

# When using build isolation, PyTorch with newer CUDA is installed and can't compile GroundingDINO
RUN python -m pip install --no-cache-dir wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python -m pip install --no-cache-dir --no-build-isolation -e GroundingDINO -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /home/appuser/Grounded-Segment-Anything/GroundingDINO
RUN pip install -r requirements.txt
RUN python setup.py install

WORKDIR /home/appuser
RUN pip install --no-cache-dir diffusers[torch]==0.15.1 opencv-python==4.7.0.72 \
    pycocotools==2.0.6 matplotlib>3.6.0 \
    onnxruntime==1.14.1 onnx==1.13.1 ipykernel==6.16.2 scipy gradio openai -i https://pypi.tuna.tsinghua.edu.cn/simple
