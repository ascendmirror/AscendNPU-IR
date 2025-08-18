FROM ubuntu:18.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="Asia/shanghai"

RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-install-suggests \
    bash \
    ca-certificates \
    wget \
    curl \
    git \
    gnupg \
    make \
    sudo \
    unzip \
    vim \
    ccache \
    aria2 \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    ca-certificates \
    clang \
    lld \
    clang-10 \
    lld-10 \
    cmake \
    ninja-build \
    patchelf \
    expect \
    python3-dev \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-ca-certificates

RUN aria2c -x 16 -s 16 https://cmake.org/files/v3.28/cmake-3.28.0-linux-x86_64.tar.gz \
    && tar -xzf cmake-3.28.0-linux-x86_64.tar.gz -C /usr/local --strip-components=1 \
    && ln -sf /usr/local/bin/cmake /usr/bin/cmake\
    && rm cmake-3.28.0-linux-x86_64.tar.gz

RUN wget https://github.com/ninja-build/ninja/releases/download/v1.12.0/ninja-linux.zip \
    && unzip ninja-linux.zip -d /usr/local/bin \
    && rm ninja-linux.zip

RUN aria2c -x 16 -s 16 https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz && \
    tar -xzf Python-3.10.0.tgz && \
    cd Python-3.10.0 && \
    ./configure \
    --enable-optimizations \
    --with-lto \
    --enable-shared \
    CC=clang \
    CXX=clang++ && \
    make -j$(nproc) && \
    make altinstall && \
    ldconfig && \
    cd .. && \
    rm -rf Python-3.10.0 Python-3.10.0.tgz

RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    python3.10 -m pip install --upgrade pip setuptools wheel

RUN python3.10 -m pip install \
    auditwheel \
    einops \
    expecttest \
    pybind11 \
    pytest \
    pytest-xdist \
    twine \
    ninja \
    patchelf \
    numpy==1.25.0

RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-10 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-10 100 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/clang-10 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++-10 100 \
    && update-alternatives --install /usr/bin/ld ld /usr/bin/ld.lld-10 100 \
    && update-alternatives --install /usr/bin/ld ld /usr/bin/ld.gold 20 \
    && update-alternatives --install /usr/bin/ld ld /usr/bin/ld.bfd 10

RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3.10 --version && \
    pip3.10 --version && \
    clang --version && \
    cmake --version && \
    ninja --version

RUN useradd bishengir \
    --create-home \
    --shell=/bin/bash \
    --uid=1000 \
    --user-group && \
    echo "bishengir ALL=(ALL) NOPASSWD:ALL" >>/etc/sudoers.d/nopasswd
	
USER bishengir

WORKDIR /home/bishengir
