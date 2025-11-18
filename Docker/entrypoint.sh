#!/bin/bash
set -e

# Sourcing ROS and workspace
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
if [ -f "/ros2_ws/install/setup.bash" ]; then
  echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc
elif [ -f "/ros2_ws/install/local_setup.bash" ]; then
  echo "source /ros2_ws/install/local_setup.bash" >> ~/.bashrc
fi

# Activate Python virtual environment
echo "source /venv/bin/activate" >> ~/.bashrc

# Execute the command that has been passed (default is bash)
exec "$@"