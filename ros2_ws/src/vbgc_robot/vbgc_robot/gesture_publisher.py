import os
import socket

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


SOCKET_PATH = "/tmp/vbgc_gesture.sock"


class GesturePublisher(Node):

    def __init__(self):
        super().__init__("gesture_publisher")

        self.publisher = self.create_publisher(
            String,
            "/gesture_command",
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
            "Gesture publisher started."
        )

        self.get_logger().info(
            f"Waiting for controller at {SOCKET_PATH}"
        )

        self.get_logger().info(
            "Publishing gestures on /gesture_command"
        )

    def check_socket(self):

        # Wait for the gesture controller to connect.
        if self.connection is None:

            try:
                self.connection, _ = self.server.accept()
                self.connection.setblocking(False)

                self.get_logger().info(
                    "Connected to gesture controller."
                )

            except BlockingIOError:
                return

        # Receive gesture data.
        try:

            data = self.connection.recv(1024)

            if not data:
                self.connection.close()
                self.connection = None
                self.buffer = ""

                self.get_logger().info(
                    "Gesture controller disconnected."
                )

                return

            self.buffer += data.decode("utf-8")

            # Process complete messages.
            while "\n" in self.buffer:

                message, self.buffer = self.buffer.split(
                    "\n",
                    1
                )

                message = message.strip()

                if message in ["FOLLOW", "STOP", "DOCK"]:

                    ros_message = String()
                    ros_message.data = message

                    self.publisher.publish(
                        ros_message
                    )

                    self.get_logger().info(
                        f"Published gesture: {message}"
                    )

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

    node = GesturePublisher()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
