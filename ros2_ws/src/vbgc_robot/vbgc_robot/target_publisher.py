import json
import os
import socket

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose


SOCKET_PATH = "/tmp/vbgc_target.sock"


class TargetPublisher(Node):

    def __init__(self):
        super().__init__("target_publisher")

        self.publisher = self.create_publisher(
            Pose,
            "/target_pose",
            10
        )

        # Remove an old socket from a previous run.
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

        # Check the socket every 10 ms.
        self.timer = self.create_timer(
            0.01,
            self.check_socket
        )

        self.get_logger().info(
            "Target publisher started."
        )

        self.get_logger().info(
            f"Waiting for vision controller at {SOCKET_PATH}"
        )

        self.get_logger().info(
            "Publishing target position on /target_pose"
        )

    def check_socket(self):

        # Wait for the vision controller to connect.
        if self.connection is None:

            try:
                self.connection, _ = self.server.accept()
                self.connection.setblocking(False)

                self.get_logger().info(
                    "Connected to vision controller."
                )

            except BlockingIOError:
                return

        # Receive target data.
        try:

            data = self.connection.recv(4096)

            if not data:
                self.connection.close()
                self.connection = None
                self.buffer = ""

                self.get_logger().info(
                    "Vision controller disconnected."
                )

                return

            self.buffer += data.decode("utf-8")

            # Process complete JSON messages.
            while "\n" in self.buffer:

                message, self.buffer = self.buffer.split(
                    "\n",
                    1
                )

                message = message.strip()

                if not message:
                    continue

                try:
                    target = json.loads(message)

                    target_x = float(target["x"])
                    target_y = float(target["y"])

                except (
                    json.JSONDecodeError,
                    KeyError,
                    TypeError,
                    ValueError
                ):

                    self.get_logger().warn(
                        f"Invalid target message: {message}"
                    )

                    continue

                pose = Pose()

                # Camera-relative target:
                # x = forward distance in metres
                # y = left/right displacement in metres
                pose.position.x = target_x
                pose.position.y = target_y
                pose.position.z = 0.0

                pose.orientation.x = 0.0
                pose.orientation.y = 0.0
                pose.orientation.z = 0.0
                pose.orientation.w = 1.0

                self.publisher.publish(pose)

        except BlockingIOError:
            pass

        except (
            ConnectionResetError,
            BrokenPipeError
        ):

            self.connection.close()
            self.connection = None
            self.buffer = ""

    def destroy_node(self):

        if self.connection is not None:
            self.connection.close()

        self.server.close()

        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = TargetPublisher()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
