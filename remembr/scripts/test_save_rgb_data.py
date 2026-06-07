#!/usr/bin/env python3
"""
Script to collect RGB images with metadata from Habitat simulation.
Subscribes to ROS2 topics and saves data in .pkl format.
"""

import os
import sys
import argparse
import time
import pickle
import cv2
import numpy as np
import threading
from datetime import datetime
from collections import deque
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CompressedImage, Image
from builtin_interfaces.msg import Time


class RGBDataCollector(Node):
    """Collects RGB images with metadata from Habitat simulation."""
    
    def __init__(self, output_dir: str, max_duration: float = None, fps: float = 1.0):
        super().__init__('rgb_data_collector')
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize state variables
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion [x, y, z, w]
        self.frame_count = 0
        
        # FPS control settings
        self.fps = fps
        self.min_frame_interval = 1.0 / fps if fps > 0 else 0  # Minimum time between saved frames (seconds)
        self.last_save_time = None  # Timestamp of last saved frame
        
        # Time limit settings
        self.max_duration = max_duration  # Maximum runtime in seconds (None = no limit)
        self.start_time = time.time()
        self.should_stop = False
        self.shutdown_initiated = False
        
        # Message buffers for synchronization (without message_filters)
        self.sync_tolerance = 0.1  # 0.1 second tolerance for synchronization
        self.rgb_buffer = deque(maxlen=10)
        self.depth_buffer = deque(maxlen=10)
        self.odom_buffer = deque(maxlen=10)
        self.lock = threading.Lock()
        
        # Subscribers (using standard ROS2 subscriptions)
        self.rgb_sub = self.create_subscription(
            CompressedImage, 
            '/habitatsim/image/head_rgb_right/compressed',
            self.rgb_callback,
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            '/habitatsim/depth/head_stereo_right_depth/image_raw',
            self.depth_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/habitatsim/platform/odom',
            self.odom_callback,
            10
        )
        
        # Timer to check for timeout (check every 1 second)
        if self.max_duration is not None:
            self.create_timer(1.0, self._check_timeout)
            self.get_logger().info(f"RGB Data Collector initialized. Max duration: {self.max_duration} seconds")
        else:
            self.get_logger().info("RGB Data Collector initialized. No time limit set.")
        
        self.get_logger().info(f"Target FPS: {self.fps} (min frame interval: {self.min_frame_interval:.3f} seconds)")
        self.get_logger().info(f"Saving to: {self.output_dir}")
        self.get_logger().info("Waiting for sensor data...")
    
    def _check_timeout(self):
        """Check if maximum duration has been exceeded."""
        if self.max_duration is None:
            return
        
        elapsed_time = time.time() - self.start_time
        remaining_time = self.max_duration - elapsed_time
        
        if remaining_time <= 0:
            self.should_stop = True
            if not self.shutdown_initiated:
                self.shutdown_initiated = True
                self.get_logger().info(f"Maximum duration ({self.max_duration}s) reached. Stopping data collection...")
                # Schedule shutdown (only once)
                self.create_timer(0.1, self._shutdown_node)
        elif remaining_time <= 10:
            # Warn when less than 10 seconds remaining
            self.get_logger().info(f"Time remaining: {remaining_time:.1f} seconds")
    
    def _shutdown_node(self):
        """Shutdown the node gracefully."""
        if not self.shutdown_initiated:
            return  # Prevent multiple calls
        self.get_logger().info("Shutting down node...")
        rclpy.shutdown()
    
    def _timestamp_to_float(self, stamp: Time) -> float:
        """Convert ROS2 Time to float (seconds since epoch)."""
        return float(stamp.sec) + float(stamp.nanosec) / 1e9
    
    def rgb_callback(self, msg: CompressedImage):
        """Callback for RGB image messages."""
        timestamp = self._timestamp_to_float(msg.header.stamp)
        with self.lock:
            self.rgb_buffer.append((timestamp, msg))
        self._try_sync_and_save()
    
    def depth_callback(self, msg: Image):
        """Callback for depth image messages."""
        timestamp = self._timestamp_to_float(msg.header.stamp)
        with self.lock:
            self.depth_buffer.append((timestamp, msg))
        self._try_sync_and_save()
    
    def odom_callback(self, msg: Odometry):
        """Callback for odometry messages."""
        timestamp = self._timestamp_to_float(msg.header.stamp)
        with self.lock:
            self.odom_buffer.append((timestamp, msg))
            # Update current position/orientation immediately (for latest state)
            self.current_position = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ])
            self.current_orientation = np.array([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ])
        self._try_sync_and_save()
    
    def _try_sync_and_save(self):
        """Try to synchronize messages from all topics and save if synchronized."""
        with self.lock:
            if len(self.rgb_buffer) == 0 or len(self.depth_buffer) == 0 or len(self.odom_buffer) == 0:
                return
            
            # Get the most recent RGB message as reference
            rgb_ts, rgb_msg = self.rgb_buffer[-1]
            
            # Find closest depth and odom messages within tolerance
            best_depth = None
            best_depth_diff = float('inf')
            for depth_ts, depth_msg in self.depth_buffer:
                diff = abs(depth_ts - rgb_ts)
                if diff < self.sync_tolerance and diff < best_depth_diff:
                    best_depth = (depth_ts, depth_msg)
                    best_depth_diff = diff
            
            best_odom = None
            best_odom_diff = float('inf')
            for odom_ts, odom_msg in self.odom_buffer:
                diff = abs(odom_ts - rgb_ts)
                if diff < self.sync_tolerance and diff < best_odom_diff:
                    best_odom = (odom_ts, odom_msg)
                    best_odom_diff = diff
            
            # If we found synchronized messages, process them
            if best_depth is not None and best_odom is not None:
                self.callback(rgb_msg, best_depth[1], best_odom[1])
    
    def callback(self, rgb_msg: CompressedImage, depth_msg: Image, odom_msg: Odometry):
        """Callback for synchronized sensor data."""
        # Check if we should stop
        if self.should_stop:
            return
        
        try:
            # Get timestamp from RGB message
            timestamp = self._timestamp_to_float(rgb_msg.header.stamp)
            
            # FPS throttling: only save frame if enough time has passed since last save
            if self.last_save_time is not None:
                time_since_last_save = timestamp - self.last_save_time
                if time_since_last_save < self.min_frame_interval:
                    # Skip this frame (too soon since last save)
                    return
            
            # Update position and orientation from odometry
            self.current_position = np.array([
                odom_msg.pose.pose.position.x,
                odom_msg.pose.pose.position.y,
                odom_msg.pose.pose.position.z
            ])
            
            self.current_orientation = np.array([
                odom_msg.pose.pose.orientation.x,
                odom_msg.pose.pose.orientation.y,
                odom_msg.pose.pose.orientation.z,
                odom_msg.pose.pose.orientation.w
            ])
            
            # Decode RGB image
            np_arr = np.frombuffer(rgb_msg.data, np.uint8)
            rgb_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if rgb_image is None:
                self.get_logger().warn("Failed to decode RGB image")
                return
            
            # Decode depth image
            depth_image = None
            if depth_msg.encoding == '32FC1':
                depth_image = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(
                    (depth_msg.height, depth_msg.width)
                )
            elif depth_msg.encoding == '16UC1':
                depth_image = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(
                    (depth_msg.height, depth_msg.width)
                )
                # Convert to meters (assuming mm units)
                depth_image = depth_image.astype(np.float32) / 1000.0
            else:
                self.get_logger().warn(f'Unsupported depth encoding: {depth_msg.encoding}')
                # Create placeholder depth
                depth_image = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.float32)
            
            # Prepare data dictionary
            # Note: 'cam0' key is required for remembr compatibility (see embedders.py)
            # Convert BGR to RGB for remembr (OpenCV uses BGR, PIL uses RGB)
            rgb_image_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            
            data = {
                'cam0': rgb_image_rgb,  # RGB format (required by remembr)
                'timestamp': timestamp,
                'position': self.current_position.copy(),  # [x, y, z] in meters
                'rotation': self.current_orientation.copy(),  # quaternion [x, y, z, w]
                'depth': depth_image,  # depth in meters
                'bbox_3d': None,  # Placeholder - 3D bounding boxes not available
                'frame_id': self.frame_count
            }
            
            # Save to pickle file using raw timestamp as filename
            # Format: frame_<unix_timestamp>.pkl
            filename = f"frame_{timestamp}.pkl"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            # Update last save time and frame count
            self.last_save_time = timestamp
            self.frame_count += 1
            
            if self.frame_count % 10 == 0:
                self.get_logger().info(f"Saved {self.frame_count} frames")
                
        except Exception as e:
            self.get_logger().error(f"Error in callback: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())


def main(args=None):
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Collect RGB images with metadata from Habitat simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run without time limit (until Ctrl+C) at default 1 fps
  python test_save_rgb_data.py
  
  # Run for 60 seconds at 1 fps (default)
  python test_save_rgb_data.py --duration 60
  
  # Run at 5 fps for 5 minutes
  python test_save_rgb_data.py -d 300 --fps 5
  
  # Run at 10 fps for 1 hour
  python test_save_rgb_data.py --duration 3600 --fps 10
  
  # Run at 0.5 fps (save every 2 seconds)
  python test_save_rgb_data.py --fps 0.5
        """
    )
    parser.add_argument(
        '-d', '--duration',
        type=float,
        default=None,
        help='Maximum runtime in seconds (default: no limit, run until Ctrl+C)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory (default: data_vlm/darpa_sim_{timestamp})'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=1.0,
        help='Target frames per second to save (default: 1.0 fps)'
    )
    
    # Parse known args (rclpy.init may add its own args)
    known_args, remaining_args = parser.parse_known_args(args)
    
    rclpy.init(args=remaining_args)
    
    # Set output directory
    if known_args.output:
        output_dir = os.path.abspath(known_args.output)
    else:
        # Default: Save to data_vlm/darpa_sim_{timestamp}
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Generate timestamp for directory name
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_name = f'darpa_sim_{timestamp_str}'
        
        # Check if we're in the mounted directory structure
        if '/root/tiamat-solution-agent' in script_dir:
            # We're in the container with mounted volume
            output_dir = os.path.join('/root/tiamat-solution-agent/agent/v2/remembr/data_vlm', dir_name)
        else:
            # Relative to script location: scripts/ -> remembr/ -> data_vlm/darpa_sim_{timestamp}
            output_dir = os.path.join(script_dir, '..', 'data_vlm', dir_name)
        output_dir = os.path.abspath(output_dir)
    
    # Create collector node
    collector = RGBDataCollector(output_dir, max_duration=known_args.duration, fps=known_args.fps)
    
    try:
        rclpy.spin(collector)
    except KeyboardInterrupt:
        collector.get_logger().info("Shutting down (KeyboardInterrupt)...")
    finally:
        elapsed_time = time.time() - collector.start_time
        collector.destroy_node()
        rclpy.shutdown()
        print(f"\n{'='*60}")
        print(f"Data collection completed!")
        print(f"Total frames saved: {collector.frame_count}")
        print(f"Total runtime: {elapsed_time:.2f} seconds")
        print(f"Target FPS: {collector.fps}")
        if collector.frame_count > 0 and elapsed_time > 0:
            actual_fps = collector.frame_count / elapsed_time
            print(f"Actual FPS: {actual_fps:.2f}")
        if collector.max_duration:
            print(f"Max duration: {collector.max_duration} seconds")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

