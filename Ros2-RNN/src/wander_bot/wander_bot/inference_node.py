import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Header
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import pickle
import os

# === Load the same features used during training ===
INPUT_FEATURES = ['x', 'y', 'lin_vel', 'ang_vel', 'cmd_lin_x', 'cmd_ang_z']
SEQ_LEN = 20

# === Encoder-Decoder model definition ===
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

# === ROS2 Node ===
class RNNPredictorNode(Node):
    def __init__(self):
        super().__init__('rnn_predictor_node')

        # Buffer for last 20 input vectors
        self.buffer = deque(maxlen=SEQ_LEN)

        # Publishers and Subscribers
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        self.pred_pub = self.create_publisher(PoseStamped, '/predicted_pose', 10)

        # Load scaler and model using correct paths
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
        except Exception as e:
            self.get_logger().error(f"Error loading model or scaler: {str(e)}")
            raise

        # Latest robot state
        self.latest_state = {key: 0.0 for key in INPUT_FEATURES}

    def odom_callback(self, msg):
        self.latest_state['x'] = msg.pose.pose.position.x
        self.latest_state['y'] = msg.pose.pose.position.y
        self.latest_state['lin_vel'] = msg.twist.twist.linear.x
        self.latest_state['ang_vel'] = msg.twist.twist.angular.z
        self.push_and_predict()

    def cmd_callback(self, msg):
        self.latest_state['cmd_lin_x'] = msg.linear.x
        self.latest_state['cmd_ang_z'] = msg.angular.z

    def push_and_predict(self):
        try:
            # Add current state to buffer
            input_vector = [self.latest_state[key] for key in INPUT_FEATURES]
            self.buffer.append(input_vector)

            # Log buffer status
            if len(self.buffer) < SEQ_LEN:
                self.get_logger().debug(f"Buffer filling: {len(self.buffer)}/{SEQ_LEN}")
                return  # Not enough data yet

            # Process input data
            input_array = np.array(self.buffer).reshape(1, SEQ_LEN, len(INPUT_FEATURES))

            # Scale input data
            try:
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

            except Exception as e:
                self.get_logger().error(f"Error during model inference: {str(e)}")
                return

            # Log prediction
            self.get_logger().info(
                f'🧠 Predicted next position: x = {prediction[0]:.2f}, y = {prediction[1]:.2f}'
            )

            # Publish as PoseStamped
            pose = PoseStamped()
            pose.header = Header()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'  # or 'odom' based on your simulation
            pose.pose.position.x = float(prediction[0])
            pose.pose.position.y = float(prediction[1])
            pose.pose.orientation.w = 1.0  # No rotation

            self.pred_pub.publish(pose)

        except Exception as e:
            self.get_logger().error(f"Unexpected error in push_and_predict: {str(e)}")
            # Continue execution despite errors

# === ROS2 Entry Point ===
def main(args=None):
    rclpy.init(args=args)
    node = RNNPredictorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
