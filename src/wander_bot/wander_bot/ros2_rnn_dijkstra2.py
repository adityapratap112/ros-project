import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from nav_msgs.msg import Path
from std_msgs.msg import Header

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import joblib
import yaml
import cv2
import os
import heapq

# === RNN Model Definition ===

class Encoder(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, hidden_size=64, output_size=2):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, hidden):
        return self.fc(hidden[-1])

class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        hidden, _ = self.encoder(x)
        output = self.decoder(hidden)
        return output

# === Main Node ===

class DijkstraRNNPlanner(Node):
    def __init__(self):
        super().__init__('ros2_rnn_dijkstra_node')

        self.map_yaml = '/home/user/ros-project/src/map.yaml'
        self.model_path = '/home/user/ros-project/src/wander_bot/wander_bot/rnn_model.pt'
        self.scaler_path = '/home/user/ros-project/src/wander_bot/wander_bot/scaler.pkl'

        self.load_map()
        self.load_model()

        self.start_pose = (10, 10)  # map coords (can be dynamic)
        self.goal_pose = None

        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.path_pub = self.create_publisher(Path, '/predicted_path', 10)

        self.get_logger().info("✅ Ready: Send goal to /goal_pose in RViz")

    def load_map(self):
        with open(self.map_yaml, 'r') as file:
            map_metadata = yaml.safe_load(file)
        img_path = self.map_yaml.replace('.yaml', '.pgm')
        self.map = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.resolution = map_metadata['resolution']
        self.origin = map_metadata['origin']

    def load_model(self):
        self.model = Seq2Seq()
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.scaler = joblib.load(self.scaler_path)

    def world_to_map(self, x, y):
        mx = int((x - self.origin[0]) / self.resolution)
        my = int((y - self.origin[1]) / self.resolution)
        return mx, my

    def map_to_world(self, mx, my):
        x = mx * self.resolution + self.origin[0]
        y = my * self.resolution + self.origin[1]
        return x, y

    def goal_callback(self, msg):
        gx, gy = msg.pose.position.x, msg.pose.position.y
        self.goal_pose = self.world_to_map(gx, gy)
        self.get_logger().info(f"🎯 Goal received: {self.goal_pose}")
        path = self.plan_path(self.start_pose, self.goal_pose)
        if path:
            rnn_path = self.predict_path(path)
            self.publish_path(rnn_path)
        else:
            self.get_logger().warn("⚠️ No path found")

    def plan_path(self, start, goal):
        h, w = self.map.shape
        visited = np.full((h, w), False)
        dist = np.full((h, w), np.inf)
        prev = np.empty((h, w), dtype=object)
        pq = [(0, start)]
        dist[start] = 0

        while pq:
            cost, current = heapq.heappop(pq)
            if visited[current]:
                continue
            visited[current] = True
            if current == goal:
                break
            for neighbor in self.get_neighbors(current, h, w):
                if self.map[neighbor] < 250:
                    continue
                new_cost = cost + 1
                if new_cost < dist[neighbor]:
                    dist[neighbor] = new_cost
                    prev[neighbor] = current
                    heapq.heappush(pq, (new_cost, neighbor))

        return self.reconstruct_path(prev, start, goal)

    def get_neighbors(self, pos, h, w):
        x, y = pos
        return [(x + dx, y + dy) for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)] if 0 <= x + dx < h and 0 <= y + dy < w]

    def reconstruct_path(self, prev, start, goal):
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = prev[current]
            if current is None:
                return []
        path.append(start)
        path.reverse()
        return [self.map_to_world(mx, my) for (mx, my) in path]

    def predict_path(self, raw_path):
        df = pd.DataFrame(raw_path, columns=["x", "y"])
        df["vx"] = df["x"].diff().fillna(0)
        df["vy"] = df["y"].diff().fillna(0)
        df["ax"] = df["vx"].diff().fillna(0)
        df["ay"] = df["vy"].diff().fillna(0)

        input_df = df[["x", "y", "vx", "vy", "ax", "ay"]]
        scaled = self.scaler.transform(input_df)
        input_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            prediction = self.model(input_tensor).numpy()[0]
        pred_coords = self.scaler.inverse_transform([prediction])[0]
        return [tuple(pred_coords)]

    def publish_path(self, path_points):
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        for x, y in path_points:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position = Point(x=x, y=y, z=0.0)
            pose.pose.orientation = Quaternion(w=1.0)
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)
        self.get_logger().info(f"✅ Published {len(path_points)} predicted points")

def main(args=None):
    rclpy.init(args=args)
    node = DijkstraRNNPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
