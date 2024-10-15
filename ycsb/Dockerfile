# Use an Ubuntu base image
FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# Install necessary tools and Python dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc-8 \
    g++-8 \
    make \
    cmake \
    libnuma-dev \
    libjemalloc-dev \
    libtcmalloc-minimal4 \
    libmysqlclient-dev \
    autoconf \
    git \
    wget \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (for Python and package management)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Set the path to conda
ENV PATH="/opt/conda/bin:$PATH"

# Copy the environment.yml file into the container
COPY environment.yml /tmp/environment.yml

# Create the environment using conda
RUN /opt/conda/bin/conda env create -f /tmp/environment.yml

# Activate the environment and ensure it's used in subsequent commands
ENV PATH /opt/conda/envs/myenv/bin:$PATH
RUN echo "source activate myenv" >> ~/.bashrc

RUN apt-get update && apt-get install -y \
    libdb++-dev \
    libdb-dev \
    && rm -rf /var/lib/apt/lists/*

RUN /opt/conda/bin/conda install -n myenv tensorboard>=1.15

# Set environment variables for GCC and G++
ENV CC=/usr/bin/gcc-8
ENV CXX=/usr/bin/g++-8

RUN apt-get update && apt-get install -y \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN pip install nevergrad

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container
COPY . .

# Set the default command to run your project and allow interaction with Python
CMD ["/bin/bash"]
