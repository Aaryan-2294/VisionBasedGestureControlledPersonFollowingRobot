import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose


class SimulatedTarget(Node):

    def __init__(self):
        super().__init__("simulated_target")

        self.publisher = self.create_publisher(
            Pose,
            "/target_pose",
            10
        )

        self.start_time = self.get_clock().now()

        self.timer = self.create_timer(
            0.05,
            self.publish_target
        )

        self.get_logger().info(
            "Simulated target started."
        )

    def publish_target(self):

        elapsed = (
            self.get_clock().now() - self.start_time
        ).nanoseconds / 1e9

        # Simulated target trajectory.
        x = 1.5 + 0.8 * math.sin(0.15 * elapsed)
        y = 0.8 * math.sin(0.30 * elapsed)

        # Direction of travel.
        dx = 0.8 * 0.15 * math.cos(0.15 * elapsed)
        dy = 0.8 * 0.30 * math.cos(0.30 * elapsed)

        theta = math.atan2(dy, dx)

        msg = Pose()

        msg.position.x = x
        msg.position.y = y

        # Store heading as a quaternion.
        msg.orientation.z = math.sin(theta / 2.0)
        msg.orientation.w = math.cos(theta / 2.0)

        self.publisher.publish(msg)


def main(args=None):

    rclpy.init(args=args)

    node = SimulatedTarget()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
