import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import pickle
import os
import math
from tf_transformations import euler_from_quaternion
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
import csv

INPUT_FEATURES = ['x', 'y', 'lin_vel', 'ang_vel', 'cmd_lin_x', 'cmd_ang_z']
SEQ_LEN = 20

# === RNN Model ===
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
    def forward(self, x):
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n

class Decoder(nn.Module):
    def __init__(self, hidden_size=64, output_size=2):
        super().__init__()
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, h):
        return self.fc(h[-1])

class Seq2One(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size, output_size=2)
    def forward(self, x):
        h_n, c_n = self.encoder(x)
        return self.decoder(h_n)

# === Controller Node ===
class RNNController(Node):
    def __init__(self):
        super().__init__('rnn_controller_node')

        # Subscribers
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pred_pub = self.create_publisher(PoseStamped, '/predicted_pose', 10)
        self.pred_path_pub = self.create_publisher(Path, '/predicted_path', 10)
        self.actual_path_pub = self.create_publisher(Path, '/actual_path', 10)

        # Load model and scaler using correct paths
        try:
            # Find the Ros2-RNN directory (repository root)
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            base_path = os.path.join(current_dir, 'Ros2-RNN')
            self.get_logger().info(f"Looking for model files in: {base_path}")

            scaler_path = os.path.join(base_path, 'scaler.pkl')
            model_path = os.path.join(base_path, 'encoder_decoder_model.pt')

            # Check if files exist
            if not os.path.exists(scaler_path):
                self.get_logger().error(f"Scaler file not found at: {scaler_path}")
                raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")

            if not os.path.exists(model_path):
                self.get_logger().error(f"Model file not found at: {model_path}")
                raise FileNotFoundError(f"Model file not found at: {model_path}")

            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
                self.get_logger().info("Successfully loaded scaler")

            self.model = Seq2One(input_size=len(INPUT_FEATURES))
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            self.get_logger().info("Successfully loaded model")

            # Update log file path
            self.log_path = os.path.join(base_path, 'rnn_control_log.csv')
        except Exception as e:
            self.get_logger().error(f"Error loading model or scaler: {str(e)}")
            raise

        # Init state
        self.latest_state = {key: 0.0 for key in INPUT_FEATURES}
        self.current_position = (0.0, 0.0)
        self.current_yaw = 0.0
        self.buffer = deque(maxlen=SEQ_LEN)
        self.pred_path = Path()
        self.actual_path = Path()
        self.obstacle_near = False

        # Log file
        try:
            self.log_path = os.path.join(base_path, 'rnn_control_log.csv')
            with open(self.log_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Time', 'Actual_x', 'Actual_y', 'Predicted_x', 'Predicted_y', 'Linear_x', 'Angular_z'])
            self.get_logger().info(f"Log file initialized at: {self.log_path}")
        except Exception as e:
            self.get_logger().error(f"Error initializing log file: {str(e)}")
            # Continue without logging if file can't be created

    def scan_callback(self, msg):
        front = msg.ranges[len(msg.ranges)//2 - 10 : len(msg.ranges)//2 + 10]
        self.obstacle_near = any(r < 0.3 for r in front if r > 0.0)
        if self.obstacle_near:
            self.get_logger().warn("🚧 Obstacle detected! Stopping.")

    def odom_callback(self, msg):
        self.current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.latest_state['x'], self.latest_state['y'] = self.current_position
        self.latest_state['lin_vel'] = msg.twist.twist.linear.x
        self.latest_state['ang_vel'] = msg.twist.twist.angular.z

        q = msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        self.append_actual_path()
        self.predict_and_control()

    def cmd_callback(self, msg):
        self.latest_state['cmd_lin_x'] = msg.linear.x
        self.latest_state['cmd_ang_z'] = msg.angular.z

    def predict_and_control(self):
        try:
            # Add current state to buffer
            input_vector = [self.latest_state[key] for key in INPUT_FEATURES]
            self.buffer.append(input_vector)

            # Check if buffer is filled
            if len(self.buffer) < SEQ_LEN:
                self.get_logger().debug(f"Buffer filling: {len(self.buffer)}/{SEQ_LEN}")
                return  # Not enough data yet

            # Process input data
            try:
                input_array = np.array(self.buffer).reshape(1, SEQ_LEN, len(INPUT_FEATURES))
                input_scaled = self.scaler.transform(input_array[0])
                input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)
            except Exception as e:
                self.get_logger().error(f"Error scaling input data: {str(e)}")
                return

            # Run model inference
            try:
                with torch.no_grad():
                    scaled_prediction = self.model(input_tensor).numpy()[0]

                # Apply inverse transform to get actual position values
                # Create a dummy array with zeros for all features except x,y (which we're predicting)
                dummy_scaled_data = np.zeros((1, len(INPUT_FEATURES) + len(['x', 'y'])))
                # The last two columns are x,y predictions
                dummy_scaled_data[0, -2:] = scaled_prediction

                # Apply inverse_transform to get the actual values
                dummy_unscaled_data = self.scaler.inverse_transform(dummy_scaled_data)
                # Extract the x,y predictions (last two columns)
                prediction = dummy_unscaled_data[0, -2:]

                pred_x, pred_y = prediction
            except Exception as e:
                self.get_logger().error(f"Error during model inference: {str(e)}")
                return

            self.get_logger().info(f'🧠 Predicted next position: x = {pred_x:.2f}, y = {pred_y:.2f}')

            # Publish PoseStamped
            try:
                pose = PoseStamped()
                pose.header.frame_id = 'odom'
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.pose.position.x = float(pred_x)
                pose.pose.position.y = float(pred_y)
                pose.pose.orientation.w = 1.0
                self.pred_pub.publish(pose)
                self.append_predicted_path(pose)
            except Exception as e:
                self.get_logger().error(f"Error publishing prediction: {str(e)}")

            # Compute control
            try:
                dx = pred_x - self.current_position[0]
                dy = pred_y - self.current_position[1]
                target_theta = math.atan2(dy, dx)
                angle_error = (target_theta - self.current_yaw + math.pi) % (2 * math.pi) - math.pi
                distance = math.hypot(dx, dy)

                twist = Twist()
                if not self.obstacle_near:
                    twist.linear.x = min(0.2, distance)
                    twist.angular.z = 1.5 * angle_error
                else:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0

                self.cmd_pub.publish(twist)
            except Exception as e:
                self.get_logger().error(f"Error computing or publishing control: {str(e)}")
                return

            # Log
            try:
                with open(self.log_path, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        self.get_clock().now().nanoseconds,
                        self.current_position[0],
                        self.current_position[1],
                        pred_x,
                        pred_y,
                        twist.linear.x,
                        twist.angular.z
                    ])
            except Exception as e:
                self.get_logger().error(f"Error writing to log file: {str(e)}")

        except Exception as e:
            self.get_logger().error(f"Unexpected error in predict_and_control: {str(e)}")
            # Continue execution despite errors

    def append_actual_path(self):
        pose = PoseStamped()
        pose.header.frame_id = 'odom'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = self.current_position[0]
        pose.pose.position.y = self.current_position[1]
        pose.pose.orientation.w = 1.0
        self.actual_path.header = pose.header
        self.actual_path.poses.append(pose)
        self.actual_path_pub.publish(self.actual_path)

    def append_predicted_path(self, pose):
        self.pred_path.header = pose.header
        self.pred_path.poses.append(pose)
        self.pred_path_pub.publish(self.pred_path)

def main(args=None):
    rclpy.init(args=args)
    node = RNNController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
