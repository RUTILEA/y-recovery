# ベースイメージとして、CUDA対応のPyTorchイメージを使用
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 非対話モードを設定し、タイムゾーンを指定
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# 必要なツールとライブラリをインストール
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
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

# pip をアップグレード
RUN python3 -m pip install --upgrade pip

# PyTorchとその依存関係をインストール
RUN pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# Detectron2をインストール
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

# Pillowを指定バージョンでインストール
RUN pip3 install pillow==9.1.0

# MatplotlibとNumpyのバージョンを指定してインストール
RUN pip3 install matplotlib numpy --upgrade

# その他の必要なPythonパッケージをインストール
RUN pip3 install opencv-python-headless pandas

# 環境変数の設定（CUDA関連）
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# デフォルトのコマンド
CMD ["/bin/bash"]