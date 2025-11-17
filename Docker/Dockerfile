# Base image with ROS 2 Jazzy installed
FROM "ros:jazzy-ros-base"

# -------- BASIC CONFIGURATION --------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    locales curl gnupg2 lsb-release build-essential git \
    python3-pip python3-venv \
    && apt autoclean -y \
    && apt autoremove -y \
    && rm -rf /var/lib/apt/lists/*


RUN apt update && apt install -y ros-jazzy-nav2-common \
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
RUN python3 -m venv /venv --system-site-packages
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
