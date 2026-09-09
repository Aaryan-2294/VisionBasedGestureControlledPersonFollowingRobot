import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class FollowingController(Node):

    def __init__(self):
        super().__init__("following_controller")

        # -----------------------------------------------------
        # Following parameters
        # -----------------------------------------------------

        self.desired_distance = 1.0
        self.distance_tolerance = 0.15

        self.max_linear_speed = 0.25
        self.max_angular_speed = 0.8

        self.target_timeout = 0.5

        # -----------------------------------------------------
        # State
        # -----------------------------------------------------

        self.follow_enabled = False
        self.target_x = None
        self.target_y = None

        self.rover_x = 0.0
        self.rover_y = 0.0
        self.rover_yaw = 0.0

        self.last_target_time = None

        # -----------------------------------------------------
        # ROS interfaces
        # -----------------------------------------------------

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            "/cmd_vel",
            10
        )

        self.gesture_subscription = self.create_subscription(
            String,
            "/gesture_command",
            self.gesture_callback,
            10
        )

        self.target_subscription = self.create_subscription(
            Pose,
            "/target_pose",
            self.target_callback,
            10
        )

        self.odom_subscription = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10
        )

        self.control_timer = self.create_timer(
            0.05,
            self.control_loop
        )

        self.get_logger().info(
            "Following controller started."
        )

        self.get_logger().info(
            "Waiting for FOLLOW / STOP / DOCK..."
        )

    # =========================================================
    # Gesture callback
    # =========================================================

    def gesture_callback(self, msg):

        command = msg.data.strip().upper()

        if command == "FOLLOW":

            self.follow_enabled = True

            self.get_logger().info(
                "FOLLOW enabled."
            )

        elif command == "STOP":

            self.follow_enabled = False

            self.publish_stop()

            self.get_logger().info(
                "STOP received. Rover stopped."
            )

        elif command == "DOCK":

            self.follow_enabled = False

            self.publish_stop()

            self.get_logger().info(
                "DOCK received. Rover stopped."
            )

    # =========================================================
    # Target callback
    # =========================================================

    def target_callback(self, msg):

        self.target_x = msg.position.x
        self.target_y = msg.position.y

        self.last_target_time = self.get_clock().now()

    # =========================================================
    # Odometry callback
    # =========================================================

    def odom_callback(self, msg):

        self.rover_x = msg.pose.pose.position.x
        self.rover_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation

        # Quaternion -> yaw
        sin_yaw = 2.0 * (
            q.w * q.z +
            q.x * q.y
        )

        cos_yaw = 1.0 - 2.0 * (
            q.y * q.y +
            q.z * q.z
        )

        self.rover_yaw = math.atan2(
            sin_yaw,
            cos_yaw
        )

    # =========================================================
    # Main control loop
    # =========================================================

    def control_loop(self):

        # No FOLLOW command.
        if not self.follow_enabled:
            self.publish_stop()
            return

        # No target available.
        if self.target_x is None or self.target_y is None:
            self.publish_stop()
            return

        # Target has stopped publishing.
        if self.last_target_time is None:
            self.publish_stop()
            return

        target_age = (
            self.get_clock().now() -
            self.last_target_time
        ).nanoseconds / 1e9

        if target_age > self.target_timeout:

            self.publish_stop()

            self.get_logger().warn(
                "Target lost. Rover stopped."
            )

            return

        # -----------------------------------------------------
        # Position difference
        # -----------------------------------------------------

        dx = self.target_x - self.rover_x
        dy = self.target_y - self.rover_y

        distance = math.sqrt(
            dx * dx +
            dy * dy
        )

        target_angle = math.atan2(
            dy,
            dx
        )

        angle_error = self.normalize_angle(
            target_angle - self.rover_yaw
        )

        # -----------------------------------------------------
        # Distance control
        # -----------------------------------------------------

        distance_error = (
            distance -
            self.desired_distance
        )

        twist = Twist()

        # -----------------------------------------------------
        # Turn toward target
        # -----------------------------------------------------

        angular_speed = 1.5 * angle_error

        angular_speed = max(
            -self.max_angular_speed,
            min(
                self.max_angular_speed,
                angular_speed
            )
        )

        twist.angular.z = angular_speed

        # -----------------------------------------------------
        # Forward movement
        # -----------------------------------------------------

        # Only move forward when reasonably aligned.
        if abs(angle_error) < 0.6:

            if distance_error > self.distance_tolerance:

                linear_speed = 0.5 * distance_error

                linear_speed = max(
                    0.0,
                    min(
                        self.max_linear_speed,
                        linear_speed
                    )
                )

                twist.linear.x = linear_speed

            else:

                twist.linear.x = 0.0

        else:

            twist.linear.x = 0.0

        self.cmd_vel_publisher.publish(
            twist
        )

    # =========================================================
    # Utility functions
    # =========================================================

    @staticmethod
    def normalize_angle(angle):

        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle

    def publish_stop(self):

        stop = Twist()

        stop.linear.x = 0.0
        stop.linear.y = 0.0
        stop.linear.z = 0.0

        stop.angular.x = 0.0
        stop.angular.y = 0.0
        stop.angular.z = 0.0

        self.cmd_vel_publisher.publish(
            stop
        )


def main(args=None):

    rclpy.init(args=args)

    node = FollowingController()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        node.publish_stop()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()
