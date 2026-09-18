import socket

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


PI_IP = "172.16.61.123"
PI_PORT = 5000


class CmdVelBridge(Node):

    def __init__(self):
        super().__init__("laptop_cmdvel_bridge")

        self.sock = None

        self.subscription = self.create_subscription(
            Twist,
            "/cmd_vel",
            self.cmd_vel_callback,
            10
        )

        self.connect_to_pi()

    def connect_to_pi(self):
        try:
            self.sock = socket.create_connection(
                (PI_IP, PI_PORT),
                timeout=5
            )
            self.get_logger().info(
                f"Connected to Pi at {PI_IP}:{PI_PORT}"
            )
        except Exception as e:
            self.get_logger().error(
                f"Could not connect to Pi: {e}"
            )

    def cmd_vel_callback(self, msg):
        if self.sock is None:
            return

        command = f"{msg.linear.x:.3f} {msg.angular.z:.3f}\n"

        try:
            self.sock.sendall(command.encode())
            self.get_logger().info(
                f"Sent: {command.strip()}"
            )
        except Exception as e:
            self.get_logger().error(
                f"Failed to send command: {e}"
            )
            self.sock = None

    def destroy_node(self):
        if self.sock is not None:
            try:
                self.sock.sendall(b"0.0 0.0\n")
                self.sock.close()
            except Exception:
                pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = CmdVelBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
