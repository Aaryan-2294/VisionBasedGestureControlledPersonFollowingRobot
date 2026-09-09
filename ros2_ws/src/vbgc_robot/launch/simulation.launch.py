import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

import xacro


def generate_launch_description():

    package_dir = get_package_share_directory("vbgc_robot")

    # ---------------------------------------------------------
    # Robot description
    # ---------------------------------------------------------

    xacro_file = os.path.join(
        package_dir,
        "description",
        "rover.urdf.xacro"
    )

    robot_description = xacro.process_file(
        xacro_file
    ).toxml()

    # ---------------------------------------------------------
    # Gazebo
    # ---------------------------------------------------------

    ros_gz_sim_dir = get_package_share_directory(
        "ros_gz_sim"
    )

    gazebo_launch = os.path.join(
        ros_gz_sim_dir,
        "launch",
        "gz_sim.launch.py"
    )

    world_file = os.path.join(
        package_dir,
        "worlds",
        "rover_world.sdf"
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            gazebo_launch
        ),
        launch_arguments={
            "gz_args": f"-r {world_file}"
        }.items()
    )

    # ---------------------------------------------------------
    # Robot State Publisher
    # ---------------------------------------------------------

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description
            }
        ]
    )

    # ---------------------------------------------------------
    # Spawn robot
    # ---------------------------------------------------------

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic",
            "robot_description",
            "-name",
            "vbgc_rover",
            "-x",
            "0",
            "-y",
            "0",
            "-z",
            "0.15"
        ],
        output="screen"
    )

    # ---------------------------------------------------------
    # ROS 2 <-> Gazebo bridge
    # ---------------------------------------------------------

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
        ],
        output="screen"
    )

    # ---------------------------------------------------------
    # Launch everything
    # ---------------------------------------------------------

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_robot,
        bridge
    ])