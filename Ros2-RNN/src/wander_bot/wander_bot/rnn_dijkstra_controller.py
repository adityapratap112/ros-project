#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RNN-Dijkstra Controller for TurtleBot3

This module implements an encoder-decoder RNN for TurtleBot3 movement control
with Dijkstra's algorithm for path planning. The implementation is designed to
be simulated in Gazebo and visualized in RViz.

Author: Autonomous Programmer
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
import torch
import torch.nn as nn
import numpy as np
from collections import deque, defaultdict
import pickle
import os
import math
import heapq
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from std_msgs.msg import Header, ColorRGBA
from builtin_interfaces.msg import Time
import csv
import time

# === Configuration ===
INPUT_FEATURES = ['x', 'y', 'lin_vel', 'ang_vel', 'cmd_lin_x', 'cmd_ang_z']
SEQ_LEN = 20
GRID_RESOLUTION = 0.1  # meters per cell
GRID_WIDTH = 200  # cells
GRID_HEIGHT = 200  # cells
OBSTACLE_THRESHOLD = 0.3  # meters
MAX_LINEAR_SPEED = 0.22  # m/s
MAX_ANGULAR_SPEED = 2.0  # rad/s

# === Encoder-Decoder RNN Model ===
class Encoder(nn.Module):
    """
    LSTM Encoder for the sequence-to-one architecture.

    Takes a sequence of robot states and encodes it into a hidden state.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        """
        Initialize the encoder.

        Args:
            input_size: Number of features in the input
            hidden_size: Size of the hidden state
            num_layers: Number of LSTM layers
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            h_n: Final hidden state
            c_n: Final cell state
        """
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n

class Decoder(nn.Module):
    """
    Linear Decoder for the sequence-to-one architecture.

    Takes the hidden state from the encoder and produces the output.
    """
    def __init__(self, hidden_size=64, output_size=2):
        """
        Initialize the decoder.

        Args:
            hidden_size: Size of the hidden state from the encoder
            output_size: Number of features in the output
        """
        super().__init__()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, h):
        """
        Forward pass through the decoder.

        Args:
            h: Hidden state from the encoder

        Returns:
            Output tensor
        """
        return self.fc(h[-1])

class Seq2One(nn.Module):
    """
    Sequence-to-One model combining the encoder and decoder.

    Takes a sequence of robot states and predicts the next position.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        """
        Initialize the sequence-to-one model.

        Args:
            input_size: Number of features in the input
            hidden_size: Size of the hidden state
            num_layers: Number of LSTM layers
        """
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers)
        self.decoder = Decoder(hidden_size, output_size=2)

    def forward(self, x):
        """
        Forward pass through the sequence-to-one model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor
        """
        h_n, c_n = self.encoder(x)
        return self.decoder(h_n)

# === Dijkstra's Algorithm Implementation ===
class DijkstraPlanner:
    """
    Path planner using Dijkstra's algorithm.

    Plans a path from the current position to a goal position
    while avoiding obstacles.
    """
    def __init__(self, resolution=GRID_RESOLUTION, width=GRID_WIDTH, height=GRID_HEIGHT):
        """
        Initialize the Dijkstra planner.

        Args:
            resolution: Grid resolution in meters per cell
            width: Grid width in cells
            height: Grid height in cells
        """
        self.resolution = resolution
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.int8)
        self.origin_x = -width * resolution / 2
        self.origin_y = -height * resolution / 2

    def update_grid(self, scan_msg, robot_x, robot_y, robot_yaw):
        """
        Update the occupancy grid based on laser scan data.

        Args:
            scan_msg: LaserScan message
            robot_x: Robot's x position
            robot_y: Robot's y position
            robot_yaw: Robot's yaw angle
        """
        # Clear the grid
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)

        # Convert robot position to grid coordinates
        robot_grid_x, robot_grid_y = self.world_to_grid(robot_x, robot_y)

        # Process each laser scan ray
        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment

        for i, distance in enumerate(scan_msg.ranges):
            # Skip invalid measurements
            if not np.isfinite(distance) or distance < scan_msg.range_min or distance > scan_msg.range_max:
                continue

            # Calculate the angle of this ray
            angle = angle_min + i * angle_increment + robot_yaw

            # Calculate the endpoint of the ray in world coordinates
            obstacle_x = robot_x + distance * math.cos(angle)
            obstacle_y = robot_y + distance * math.sin(angle)

            # Convert to grid coordinates
            obstacle_grid_x, obstacle_grid_y = self.world_to_grid(obstacle_x, obstacle_y)

            # Check if the coordinates are within the grid
            if 0 <= obstacle_grid_x < self.width and 0 <= obstacle_grid_y < self.height:
                # Mark the cell as occupied
                self.grid[obstacle_grid_y, obstacle_grid_x] = 100

                # Inflate obstacles for safety
                self.inflate_obstacle(obstacle_grid_x, obstacle_grid_y, int(OBSTACLE_THRESHOLD / self.resolution))

    def inflate_obstacle(self, x, y, radius):
        """
        Inflate an obstacle to ensure the robot keeps a safe distance.

        Args:
            x: Obstacle x coordinate in grid
            y: Obstacle y coordinate in grid
            radius: Inflation radius in cells
        """
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        self.grid[ny, nx] = 100

    def world_to_grid(self, x, y):
        """
        Convert world coordinates to grid coordinates.

        Args:
            x: X coordinate in world frame
            y: Y coordinate in world frame

        Returns:
            grid_x, grid_y: Coordinates in grid frame
        """
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """
        Convert grid coordinates to world coordinates.

        Args:
            grid_x: X coordinate in grid frame
            grid_y: Y coordinate in grid frame

        Returns:
            x, y: Coordinates in world frame
        """
        x = grid_x * self.resolution + self.origin_x
        y = grid_y * self.resolution + self.origin_y
        return x, y

    def plan_path(self, start_x, start_y, goal_x, goal_y):
        """
        Plan a path from start to goal using Dijkstra's algorithm.

        Args:
            start_x: Start x position in world coordinates
            start_y: Start y position in world coordinates
            goal_x: Goal x position in world coordinates
            goal_y: Goal y position in world coordinates

        Returns:
            path: List of (x, y) coordinates in world frame
        """
        # Convert to grid coordinates
        start_grid_x, start_grid_y = self.world_to_grid(start_x, start_y)
        goal_grid_x, goal_grid_y = self.world_to_grid(goal_x, goal_y)

        # Check if start or goal is out of bounds or in an obstacle
        if (start_grid_x < 0 or start_grid_x >= self.width or 
            start_grid_y < 0 or start_grid_y >= self.height or
            goal_grid_x < 0 or goal_grid_x >= self.width or
            goal_grid_y < 0 or goal_grid_y >= self.height):
            return []

        if self.grid[start_grid_y, start_grid_x] == 100 or self.grid[goal_grid_y, goal_grid_x] == 100:
            return []

        # Initialize Dijkstra's algorithm
        start = (start_grid_x, start_grid_y)
        goal = (goal_grid_x, goal_grid_y)

        # Priority queue for Dijkstra's algorithm
        queue = [(0, start)]
        heapq.heapify(queue)

        # Dictionary to store distances
        dist = {start: 0}

        # Dictionary to store predecessors
        prev = {}

        # Directions for 8-connected grid
        directions = [
            (1, 0), (0, 1), (-1, 0), (0, -1),  # 4-connected
            (1, 1), (-1, 1), (-1, -1), (1, -1)  # Diagonals
        ]

        # Dijkstra's algorithm
        while queue:
            current_dist, current = heapq.heappop(queue)

            # If we reached the goal, reconstruct the path
            if current == goal:
                path = []
                while current in prev:
                    x, y = self.grid_to_world(current[0], current[1])
                    path.append((x, y))
                    current = prev[current]

                # Add the start position
                x, y = self.grid_to_world(start[0], start[1])
                path.append((x, y))

                # Reverse the path to get start-to-goal order
                path.reverse()
                return path

            # If we've already found a better path to this node, skip it
            if current_dist > dist[current]:
                continue

            # Check all neighbors
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                # Check if the neighbor is valid
                if (neighbor[0] < 0 or neighbor[0] >= self.width or
                    neighbor[1] < 0 or neighbor[1] >= self.height or
                    self.grid[neighbor[1], neighbor[0]] == 100):
                    continue

                # Calculate the distance to the neighbor
                if dx == 0 or dy == 0:
                    # Horizontal or vertical movement
                    new_dist = current_dist + 1
                else:
                    # Diagonal movement
                    new_dist = current_dist + 1.414  # sqrt(2)

                # If we found a better path to the neighbor, update it
                if neighbor not in dist or new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    prev[neighbor] = current
                    heapq.heappush(queue, (new_dist, neighbor))

        # If we get here, there's no path to the goal
        return []

# === RNN-Dijkstra Controller Node ===
class RNNDijkstraController(Node):
    """
    ROS2 node that combines an encoder-decoder RNN with Dijkstra's algorithm
    for TurtleBot3 movement control and path planning.
    """
    def __init__(self):
        """Initialize the RNN-Dijkstra controller node."""
        super().__init__('rnn_dijkstra_controller')

        # Initialize the Dijkstra planner
        self.planner = DijkstraPlanner()

        # Subscribers
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pred_pub = self.create_publisher(PoseStamped, '/predicted_pose', 10)
        self.pred_path_pub = self.create_publisher(Path, '/predicted_path', 10)
        self.actual_path_pub = self.create_publisher(Path, '/actual_path', 10)
        self.planned_path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/occupancy_grid', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/visualization_markers', 10)

        # Load model and scaler
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
            self.log_path = os.path.join(base_path, 'rnn_dijkstra_control_log.csv')
        except Exception as e:
            self.get_logger().error(f"Error loading model or scaler: {str(e)}")
            raise

        # Initialize state variables
        self.latest_state = {key: 0.0 for key in INPUT_FEATURES}
        self.current_position = (0.0, 0.0)
        self.current_yaw = 0.0
        self.buffer = deque(maxlen=SEQ_LEN)
        self.pred_path = Path()
        self.actual_path = Path()
        self.planned_path = Path()
        self.obstacle_near = False
        self.scan_data = None
        self.goal_position = None

        # Initialize the log file
        try:
            with open(self.log_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Time', 'Actual_x', 'Actual_y', 'Predicted_x', 'Predicted_y', 
                    'Goal_x', 'Goal_y', 'Linear_x', 'Angular_z'
                ])
            self.get_logger().info(f"Log file initialized at: {self.log_path}")
        except Exception as e:
            self.get_logger().error(f"Error initializing log file: {str(e)}")
            # Continue without logging if file can't be created

        # Create a timer for the control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info("RNN-Dijkstra Controller initialized")

    def scan_callback(self, msg):
        """
        Process laser scan data to detect obstacles.

        Args:
            msg: LaserScan message
        """
        # Store the scan data for grid updates
        self.scan_data = msg

        # Check for obstacles in front of the robot
        front_indices = range(len(msg.ranges) // 2 - 10, len(msg.ranges) // 2 + 10)
        front_ranges = [msg.ranges[i] for i in front_indices if np.isfinite(msg.ranges[i])]

        self.obstacle_near = any(r < OBSTACLE_THRESHOLD for r in front_ranges if r > 0.0)

        if self.obstacle_near:
            self.get_logger().warn("🚧 Obstacle detected! Adjusting path.")

    def odom_callback(self, msg):
        """
        Process odometry data to update the robot's state.

        Args:
            msg: Odometry message
        """
        # Update position and velocity
        self.current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.latest_state['x'], self.latest_state['y'] = self.current_position
        self.latest_state['lin_vel'] = msg.twist.twist.linear.x
        self.latest_state['ang_vel'] = msg.twist.twist.angular.z

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # Update the actual path
        self.append_actual_path()

        # Update the input buffer for the RNN
        input_vector = [self.latest_state[key] for key in INPUT_FEATURES]
        self.buffer.append(input_vector)

    def cmd_callback(self, msg):
        """
        Process command velocity data.

        Args:
            msg: Twist message
        """
        self.latest_state['cmd_lin_x'] = msg.linear.x
        self.latest_state['cmd_ang_z'] = msg.angular.z

    def predict_next_position(self):
        """
        Use the RNN model to predict the next position.

        Returns:
            pred_x, pred_y: Predicted x and y coordinates
        """
        if len(self.buffer) < SEQ_LEN:
            self.get_logger().debug(f"Buffer filling: {len(self.buffer)}/{SEQ_LEN}")
            return None, None

        try:
            # Process input data
            input_array = np.array(self.buffer).reshape(1, SEQ_LEN, len(INPUT_FEATURES))
            input_scaled = self.scaler.transform(input_array[0])
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0)

            # Run model inference
            with torch.no_grad():
                scaled_prediction = self.model(input_tensor).numpy()[0]

            # Apply inverse transform to get actual position values
            dummy_scaled_data = np.zeros((1, len(INPUT_FEATURES) + len(['x', 'y'])))
            dummy_scaled_data[0, -2:] = scaled_prediction
            dummy_unscaled_data = self.scaler.inverse_transform(dummy_scaled_data)
            prediction = dummy_unscaled_data[0, -2:]

            pred_x, pred_y = prediction

            # Publish the predicted pose
            pose = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(pred_x)
            pose.pose.position.y = float(pred_y)
            pose.pose.orientation.w = 1.0
            self.pred_pub.publish(pose)
            self.append_predicted_path(pose)

            self.get_logger().info(f'🧠 Predicted next position: x = {pred_x:.2f}, y = {pred_y:.2f}')

            return pred_x, pred_y

        except Exception as e:
            self.get_logger().error(f"Error in predict_next_position: {str(e)}")
            return None, None

    def update_occupancy_grid(self):
        """Update the occupancy grid based on laser scan data."""
        if self.scan_data is None:
            return

        try:
            # Update the grid in the planner
            self.planner.update_grid(self.scan_data, 
                                    self.current_position[0], 
                                    self.current_position[1], 
                                    self.current_yaw)

            # Publish the occupancy grid for visualization
            grid_msg = OccupancyGrid()
            grid_msg.header.frame_id = 'odom'
            grid_msg.header.stamp = self.get_clock().now().to_msg()
            grid_msg.info.resolution = self.planner.resolution
            grid_msg.info.width = self.planner.width
            grid_msg.info.height = self.planner.height
            grid_msg.info.origin.position.x = self.planner.origin_x
            grid_msg.info.origin.position.y = self.planner.origin_y
            grid_msg.info.origin.orientation.w = 1.0

            # Flatten the grid for the message
            grid_msg.data = self.planner.grid.flatten().tolist()

            self.grid_pub.publish(grid_msg)

        except Exception as e:
            self.get_logger().error(f"Error updating occupancy grid: {str(e)}")

    def plan_path_to_goal(self, goal_x, goal_y):
        """
        Plan a path to the goal using Dijkstra's algorithm.

        Args:
            goal_x: Goal x coordinate
            goal_y: Goal y coordinate

        Returns:
            path: List of (x, y) coordinates
        """
        try:
            # Update the occupancy grid
            self.update_occupancy_grid()

            # Plan the path
            path = self.planner.plan_path(
                self.current_position[0], 
                self.current_position[1],
                goal_x, 
                goal_y
            )

            if not path:
                self.get_logger().warn("No path found to goal!")
                return []

            # Publish the planned path
            planned_path = Path()
            planned_path.header.frame_id = 'odom'
            planned_path.header.stamp = self.get_clock().now().to_msg()

            for x, y in path:
                pose = PoseStamped()
                pose.header = planned_path.header
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.orientation.w = 1.0
                planned_path.poses.append(pose)

            self.planned_path = planned_path
            self.planned_path_pub.publish(planned_path)

            # Visualize the path with markers
            self.visualize_path(path)

            return path

        except Exception as e:
            self.get_logger().error(f"Error in plan_path_to_goal: {str(e)}")
            return []

    def visualize_path(self, path):
        """
        Visualize the planned path with markers.

        Args:
            path: List of (x, y) coordinates
        """
        if not path:
            return

        marker_array = MarkerArray()

        # Line strip for the path
        line_marker = Marker()
        line_marker.header.frame_id = 'odom'
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = 'path'
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.05  # Line width
        line_marker.color.r = 0.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0
        line_marker.pose.orientation.w = 1.0

        for x, y in path:
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.05  # Slightly above ground
            line_marker.points.append(point)

        marker_array.markers.append(line_marker)

        # Spheres for waypoints
        for i, (x, y) in enumerate(path):
            sphere_marker = Marker()
            sphere_marker.header.frame_id = 'odom'
            sphere_marker.header.stamp = self.get_clock().now().to_msg()
            sphere_marker.ns = 'waypoints'
            sphere_marker.id = i + 1
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            sphere_marker.pose.position.x = x
            sphere_marker.pose.position.y = y
            sphere_marker.pose.position.z = 0.05
            sphere_marker.pose.orientation.w = 1.0
            sphere_marker.scale.x = 0.1
            sphere_marker.scale.y = 0.1
            sphere_marker.scale.z = 0.1

            # Color gradient from green (start) to red (goal)
            t = i / max(1, len(path) - 1)
            sphere_marker.color.r = t
            sphere_marker.color.g = 1.0 - t
            sphere_marker.color.b = 0.0
            sphere_marker.color.a = 1.0

            marker_array.markers.append(sphere_marker)

        self.marker_pub.publish(marker_array)

    def control_loop(self):
        """Main control loop that runs at a fixed frequency."""
        try:
            # Predict the next position using the RNN
            pred_x, pred_y = self.predict_next_position()

            if pred_x is None or pred_y is None:
                return

            # Use the predicted position as the goal
            self.goal_position = (pred_x, pred_y)

            # Plan a path to the goal
            path = self.plan_path_to_goal(pred_x, pred_y)

            # If no path is found, try to move directly to the goal
            if not path:
                self.get_logger().warn("No path found, attempting direct movement")
                self.move_to_goal(pred_x, pred_y)
                return

            # Follow the path
            self.follow_path(path)

            # Log data
            self.log_data(pred_x, pred_y)

        except Exception as e:
            self.get_logger().error(f"Error in control_loop: {str(e)}")

    def follow_path(self, path):
        """
        Follow the planned path.

        Args:
            path: List of (x, y) coordinates
        """
        if not path or len(path) < 2:
            return

        # Get the next waypoint to follow
        # We use the second point in the path as the immediate goal
        # (the first point is the current position)
        next_x, next_y = path[1]

        # Move to the next waypoint
        self.move_to_goal(next_x, next_y)

    def move_to_goal(self, goal_x, goal_y):
        """
        Generate velocity commands to move toward a goal position.

        Args:
            goal_x: Goal x coordinate
            goal_y: Goal y coordinate
        """
        try:
            # Calculate the direction and distance to the goal
            dx = goal_x - self.current_position[0]
            dy = goal_y - self.current_position[1]

            # Calculate the angle to the goal
            target_angle = math.atan2(dy, dx)

            # Calculate the angle error (normalized to [-pi, pi])
            angle_error = (target_angle - self.current_yaw + math.pi) % (2 * math.pi) - math.pi

            # Calculate the distance to the goal
            distance = math.hypot(dx, dy)

            # Create a Twist message for the velocity command
            twist = Twist()

            # If there's an obstacle nearby, stop and rotate
            if self.obstacle_near:
                twist.linear.x = 0.0
                twist.angular.z = 0.5  # Rotate to find a clear path
                self.get_logger().warn("Obstacle near, stopping and rotating")
            else:
                # Set linear velocity proportional to distance, with a maximum
                twist.linear.x = min(MAX_LINEAR_SPEED, 0.5 * distance)

                # Set angular velocity proportional to angle error, with a maximum
                twist.angular.z = min(MAX_ANGULAR_SPEED, 
                                     max(-MAX_ANGULAR_SPEED, 1.5 * angle_error))

                self.get_logger().info(
                    f"Moving to goal: distance={distance:.2f}, angle_error={angle_error:.2f}, "
                    f"linear={twist.linear.x:.2f}, angular={twist.angular.z:.2f}"
                )

            # Publish the velocity command
            self.cmd_pub.publish(twist)

            return twist

        except Exception as e:
            self.get_logger().error(f"Error in move_to_goal: {str(e)}")
            return None

    def append_actual_path(self):
        """Add the current position to the actual path and publish it."""
        try:
            pose = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = self.current_position[0]
            pose.pose.position.y = self.current_position[1]

            # Set orientation from yaw
            q = quaternion_from_euler(0, 0, self.current_yaw)
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]

            # Add to path
            self.actual_path.header = pose.header
            self.actual_path.poses.append(pose)

            # Limit the path length to avoid excessive memory usage
            if len(self.actual_path.poses) > 1000:
                self.actual_path.poses.pop(0)

            # Publish the path
            self.actual_path_pub.publish(self.actual_path)

        except Exception as e:
            self.get_logger().error(f"Error in append_actual_path: {str(e)}")

    def append_predicted_path(self, pose):
        """
        Add a predicted pose to the predicted path and publish it.

        Args:
            pose: PoseStamped message
        """
        try:
            self.pred_path.header = pose.header
            self.pred_path.poses.append(pose)

            # Limit the path length to avoid excessive memory usage
            if len(self.pred_path.poses) > 1000:
                self.pred_path.poses.pop(0)

            # Publish the path
            self.pred_path_pub.publish(self.pred_path)

        except Exception as e:
            self.get_logger().error(f"Error in append_predicted_path: {str(e)}")

    def log_data(self, pred_x, pred_y):
        """
        Log data to a CSV file.

        Args:
            pred_x: Predicted x coordinate
            pred_y: Predicted y coordinate
        """
        try:
            # Get the current velocity command
            twist = self.move_to_goal(pred_x, pred_y)

            if twist is None:
                return

            # Write to the log file
            with open(self.log_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.get_clock().now().nanoseconds,
                    self.current_position[0],
                    self.current_position[1],
                    pred_x,
                    pred_y,
                    self.goal_position[0] if self.goal_position else 0.0,
                    self.goal_position[1] if self.goal_position else 0.0,
                    twist.linear.x,
                    twist.angular.z
                ])

        except Exception as e:
            self.get_logger().error(f"Error in log_data: {str(e)}")

def main(args=None):
    """ROS2 entry point."""
    rclpy.init(args=args)
    node = RNNDijkstraController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
