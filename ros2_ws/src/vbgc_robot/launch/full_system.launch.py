import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

import xacro


def generate_launch_description():

    package_dir = get_package_share_directory(
        "vbgc_robot"
    )

    # --------------------------------------------------------
    # Robot description
    # --------------------------------------------------------

    xacro_file = os.path.join(
        package_dir,
        "description",
        "rover.urdf.xacro"
    )

    robot_description = xacro.process_file(
        xacro_file
    ).toxml()

    # --------------------------------------------------------
    # Gazebo
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Robot State Publisher
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Spawn rover
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Gazebo <-> ROS 2 bridge
    # --------------------------------------------------------

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
        ],
        output="screen"
    )

    # --------------------------------------------------------
    # Gesture publisher
    # --------------------------------------------------------

    gesture_publisher = Node(
        package="vbgc_robot",
        executable="gesture_publisher",
        output="screen"
    )

    # --------------------------------------------------------
    # Camera target publisher
    # --------------------------------------------------------

    camera_target_publisher = Node(
        package="vbgc_robot",
        executable="camera_target_publisher",
        output="screen"
    )

    # --------------------------------------------------------
    # Following controller
    # --------------------------------------------------------

    following_controller = Node(
        package="vbgc_robot",
        executable="following_controller",
        output="screen"
    )

    # --------------------------------------------------------
    # Start ROS 2 simulation system
    # --------------------------------------------------------

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_robot,
        bridge,
        gesture_publisher,
        camera_target_publisher,
        following_controller
    ])