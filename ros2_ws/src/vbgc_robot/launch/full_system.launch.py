import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

import xacro


def generate_launch_description():

    # ========================================================
    # PATHS
    # ========================================================

    package_dir = get_package_share_directory(
        "vbgc_robot"
    )

    # IMPORTANT:
    # cc_ll.py uses relative paths for:
    # yolo11n.pt
    # hand_landmarker.task
    # gesture_data/
    # gesture_model.pkl
    #
    # Therefore its working directory MUST be the repository root.
    repo_dir = os.path.expanduser(
        "~/robot_vision/vbgcRobot"
    )

    vision_python = os.path.expanduser(
        "~/robot_vision/.venv/bin/python3"
    )

    cc_ll_path = os.path.join(
        repo_dir,
        "cc_ll.py"
    )

    # ========================================================
    # ROBOT DESCRIPTION
    # ========================================================

    xacro_file = os.path.join(
        package_dir,
        "description",
        "rover.urdf.xacro"
    )

    robot_description = xacro.process_file(
        xacro_file
    ).toxml()

    # ========================================================
    # GAZEBO
    # ========================================================

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

    # ========================================================
    # ROBOT STATE PUBLISHER
    # ========================================================

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

    # ========================================================
    # SPAWN ROVER IN GAZEBO
    # ========================================================

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

    # ========================================================
    # GAZEBO <-> ROS 2 BRIDGE
    # ========================================================

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
        ],
        output="screen"
    )

    # ========================================================
    # GESTURE PUBLISHER
    # ========================================================

    gesture_publisher = Node(
        package="vbgc_robot",
        executable="gesture_publisher",
        output="screen"
    )

    # ========================================================
    # CAMERA TARGET PUBLISHER
    # ========================================================

    camera_target_publisher = Node(
        package="vbgc_robot",
        executable="camera_target_publisher",
        output="screen"
    )

    # ========================================================
    # FOLLOWING CONTROLLER
    # ========================================================

    following_controller = Node(
        package="vbgc_robot",
        executable="following_controller",
        output="screen"
    )

    # ========================================================
    # ESP32 MOTOR BRIDGE
    # ========================================================

    esp32_motor_bridge = Node(
        package="vbgc_robot",
        executable="esp32_motor_bridge",
        output="screen",
        parameters=[
            {
                "serial_port": "/dev/ttyUSB0",
                "baud_rate": 115200,
                "command_timeout": 0.5
            }
        ]
    )

    # ========================================================
    # CC_LL VISION + GESTURE CONTROLLER
    # ========================================================

    cc_ll = ExecuteProcess(
        cmd=[
            vision_python,
            cc_ll_path
        ],
        cwd=repo_dir,
        output="screen"
    )

    # ========================================================
    # START EVERYTHING
    # ========================================================

    return LaunchDescription([

        # Gazebo
        gazebo,
        robot_state_publisher,
        spawn_robot,
        bridge,

        # ROS control
        gesture_publisher,
        camera_target_publisher,
        following_controller,

        # Physical rover
        esp32_motor_bridge,

        # Camera + person tracking + gesture recognition
        cc_ll,
    ])
