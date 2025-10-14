#!/usr/bin/env python3

"""Launches a langgraph agent as an ros action server node with default parameters."""

import os

from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    venv_python = os.path.join(os.environ['VIRTUAL_ENV'], 'bin', 'python')
    # Getting directories and launch-files
    langgraph_ros_dir = get_package_share_directory('langgraph_ros')
    default_params_file = os.path.join(langgraph_ros_dir, 'params', 'default_params.yaml')
    default_mcp_servers_file = os.path.join(langgraph_ros_dir, 'params', 'mcp.json')
    default_sys_prompt_file = os.path.join(langgraph_ros_dir, 'templates', 'system_prompt.jinja')

    # Input parameters declaration
    params_file = LaunchConfiguration('params_file')
    mcp_servers_file = LaunchConfiguration('mcp_servers_file')
    system_prompt_file = LaunchConfiguration('system_prompt_file')
    log_level = LaunchConfiguration('log-level')

    declare_params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Full path to the ROS2 parameters file with configuration'
    )

    declare_mcp_servers_file_arg = DeclareLaunchArgument(
        'mcp_servers_file',
        default_value=default_mcp_servers_file,
        description='Full path to the MCP servers configuration file'
    )

    declare_system_prompt_file_arg = DeclareLaunchArgument(
        'system_prompt_file',
        default_value=default_sys_prompt_file,
        description='Full path to the system prompt template file'
    )

    declare_log_level_arg = DeclareLaunchArgument(
        name='log-level',
        default_value='info',
        description='Logging level (info, debug, ...)'
    )

    # Create our own temporary YAML files that include substitutions
    param_substitutions = {
        'mcp_servers': mcp_servers_file,
        'system_prompt': system_prompt_file,
    }

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key='',
        param_rewrites=param_substitutions,
        convert_types=True
    )

    # Prepare the detection node.
    langgraph_ros_node = Node(
        package='langgraph_ros',
        executable='use_case_graph_ros',
        name='langgraph_action_server',
        output='screen',
        prefix=[venv_python, ' -u '],
        parameters=[configured_params],
        arguments=['--ros-args', '--log-level', log_level],
    )

    return LaunchDescription([
        declare_params_file_arg,
        declare_mcp_servers_file_arg,
        declare_system_prompt_file_arg,
        declare_log_level_arg,
        langgraph_ros_node,
    ])
