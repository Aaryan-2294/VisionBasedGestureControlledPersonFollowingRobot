import serial

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class ESP32MotorBridge(Node):

    def __init__(self):
        super().__init__('esp32_motor_bridge')

        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('command_timeout', 0.5)

        self.serial_port = self.get_parameter(
            'serial_port'
        ).get_parameter_value().string_value

        self.baud_rate = self.get_parameter(
            'baud_rate'
        ).get_parameter_value().integer_value

        self.command_timeout = self.get_parameter(
            'command_timeout'
        ).get_parameter_value().double_value

        try:
            self.serial_connection = serial.Serial(
                self.serial_port,
                self.baud_rate,
                timeout=1
            )

            self.get_logger().info(
                f'Connected to ESP32 on {self.serial_port} '
                f'at {self.baud_rate} baud.'
            )

        except serial.SerialException as error:
            self.serial_connection = None

            self.get_logger().error(
                f'Could not open {self.serial_port}: {error}'
            )

        self.last_command_time = self.get_clock().now()

        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        self.safety_timer = self.create_timer(
            0.1,
            self.safety_check
        )

    def send_command(self, linear, angular):

        if self.serial_connection is None:
            return

        command = f'{linear:.3f} {angular:.3f}\n'

        try:
            self.serial_connection.write(
                command.encode('utf-8')
            )

        except serial.SerialException as error:
            self.get_logger().error(
                f'Failed to send command to ESP32: {error}'
            )

    def cmd_vel_callback(self, msg):

        linear = msg.linear.x
        angular = msg.angular.z

        self.last_command_time = self.get_clock().now()

        self.send_command(linear, angular)

    def safety_check(self):

        elapsed = (
            self.get_clock().now() -
            self.last_command_time
        ).nanoseconds / 1e9

        if elapsed > self.command_timeout:
            self.send_command(0.0, 0.0)

    def destroy_node(self):

        if self.serial_connection is not None:
            try:
                self.send_command(0.0, 0.0)
                self.serial_connection.close()
            except serial.SerialException:
                pass

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = ESP32MotorBridge()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
