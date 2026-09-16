import json
import os
import socket

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose


SOCKET_PATH = "/tmp/vbgc_target.sock"


class CameraTargetPublisher(Node):

    def __init__(self):
        super().__init__("camera_target_publisher")

        self.publisher = self.create_publisher(
            Pose,
            "/target_pose",
            10
        )

        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

        self.server = socket.socket(
            socket.AF_UNIX,
            socket.SOCK_STREAM
        )

        self.server.bind(SOCKET_PATH)
        self.server.listen(1)
        self.server.setblocking(False)

        self.connection = None
        self.buffer = ""

        self.timer = self.create_timer(
            0.01,
            self.check_socket
        )

        self.get_logger().info(
            "Camera target publisher started."
        )
        self.get_logger().info(
            f"Waiting for camera controller at {SOCKET_PATH}"
        )
        self.get_logger().info(
            "Publishing camera-relative target on /target_pose"
        )

    def check_socket(self):

        if self.connection is None:
            try:
                self.connection, _ = self.server.accept()
                self.connection.setblocking(False)
            except BlockingIOError:
                return

        try:
            data = self.connection.recv(4096)

            if not data:
                self.close_connection()
                return

            self.buffer += data.decode("utf-8")

            while "\n" in self.buffer:
                message, self.buffer = self.buffer.split("\n", 1)
                message = message.strip()

                if not message:
                    continue

                self.publish_target(message)

        except BlockingIOError:
            pass
        except (
            ConnectionResetError,
            BrokenPipeError,
            UnicodeDecodeError,
            OSError,
        ):
            self.close_connection()

    def publish_target(self, message):

        try:
            target = json.loads(message)
            x = float(target["x"])
            y = float(target["y"])
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            self.get_logger().warn(
                "Ignoring invalid camera target message."
            )
            return

        msg = Pose()
        msg.position.x = x
        msg.position.y = y
        msg.position.z = 0.0

        # Target orientation is not required for following.
        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = 0.0
        msg.orientation.w = 1.0

        self.publisher.publish(msg)

    def close_connection(self):

        if self.connection is not None:
            self.connection.close()

        self.connection = None
        self.buffer = ""

    def destroy_node(self):

        self.close_connection()
        self.server.close()

        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = CameraTargetPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
