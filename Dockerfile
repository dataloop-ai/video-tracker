ARG AGNT_VERSION

FROM gcr.io/viewo-g/piper/agent/runner/gpu/main:$AGNT_VERSION
MAINTAINER Or Shabtay <or@dataloop.ai>

# Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir \
    torch==1.1.0 \
    opencv_python==4.1.2.30 \
    matplotlib==2.1.2 \
    torchvision


