# Base image with CUDA and Ubuntu 24.04
FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

# -------- BASIC CONFIGURATION --------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    locales curl gnupg2 lsb-release build-essential git \
    python3-pip python3-venv \
    && apt autoclean -y \
    && apt autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Configure locale
RUN locale-gen en_US en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# -------- INSTALL ROS 2 JAZZY --------
# Install ROS
ENV ROS_DISTRO jazzy
# desktop or ros-base
ARG INSTALL_PACKAGE=desktop

RUN apt update -q && \
    apt install -y curl gnupg2 lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt update -q && \
    apt install -y ros-${ROS_DISTRO}-${INSTALL_PACKAGE} \
    python3-argcomplete \
    python3-colcon-common-extensions \
    python3-rosdep python3-vcstool && \
    rosdep init && \
    rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y ros-${ROS_DISTRO}-nav2-common \
    && apt autoclean -y \
    && apt autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# -------- CONFIGURE WORKSPACE --------
WORKDIR /ros2_ws/src

# Clone dependencies
RUN git clone https://github.com/grupo-avispa/llm_interactions_msgs.git
RUN git clone https://github.com/grupo-avispa/langgraph_base_ros.git

# Copy the entire package
COPY ./ /ros2_ws/src/pisito_agent/

WORKDIR /ros2_ws

# Create Python virtual environment
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install Python dependencies
WORKDIR /ros2_ws/src/pisito_agent
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install empy catkin_pkg lark

# Compile the package
WORKDIR /ros2_ws
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && colcon build 

# -------- ENTRYPOINT --------
# Copy the entrypoint script that will automatically "source" the environment
COPY Docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
