# ベースイメージとして、CUDA対応のPyTorchイメージを使用
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# ベースイメージとしてPython 3.9を使用
FROM python:3.9

# 非対話モードを設定し、タイムゾーンを指定
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# 必要なツールとライブラリをインストール
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    wget \
    git \
    tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# タイムゾーンの設定を非対話的に行う
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# 作業ディレクトリを作成
WORKDIR /workspace

# PyTorchとDetectron2をインストール
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118


# Detectron2をインストール
# RUN pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.3.1/index.html
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

# Pillowを最新バージョンにアップグレード
RUN pip3 install pillow==9.1.0

# MatplotlibとNumpyのバージョンを指定してインストール
RUN pip3 install matplotlib numpy --upgrade

# その他の必要なPythonパッケージをインストール
RUN pip3 install opencv-python-headless pandas 


