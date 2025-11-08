import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int16, Bool
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

from ros2_sam.sam_client import SAMClient
from ros2_sam.utils import show_box, show_mask, show_points

class AriaToSAM(Node):
    def __init__(self):
        super().__init__("sam_listener")

        self.image_msg = None
        self.gaze_x = None
        self.gaze_y = None
        self.bridge = CvBridge()

        # SAM client
        self.sam_client = SAMClient(
            node_name="sam_client_wrapper",
            service_name="sam_server/segment",
        )

        # Subscribers
        self.create_subscription(Image, '/rgb/image_rect', self.image_cb, 10)
        self.create_subscription(Int16, '/gaze/x_rgb', self.gaze_x_cb, 10)
        self.create_subscription(Int16, '/gaze/y_rgb', self.gaze_y_cb, 10)
        self.create_subscription(Bool, '/capture_trigger', self.trigger_cb, 10)
        self.get_logger().info("AriaToSAM initialized — publish 'true' to /capture_trigger to capture image and gaze")

        # Publisher
        self.image_pub = self.create_publisher(Image, "/sam/cropped_image", 10)
        self.gaze_pub = self.create_publisher(Point, "/sam/gaze_point", 10)

    def trigger_cb(self, msg: Bool):
        """Called when a trigger message arrives."""
        if msg.data:
            self.get_logger().info("Trigger received! Capturing image and gaze...")
            if self.image_msg is not None and self.gaze_x is not None and self.gaze_y is not None:
                self.get_logger().info("All data ready — processing capture")
                self.process_image(self.image_msg, self.gaze_x, self.gaze_y)
            else:
                self.get_logger().info("Data missing!")

    # ---------- Callbacks ----------
    def image_cb(self, msg):
        self.image_msg = msg

    def gaze_x_cb(self, msg):
        self.gaze_x = msg.data

    def gaze_y_cb(self, msg):
        self.gaze_y = msg.data

    # ---------- Main logic ----------
    def process_image(self, image_msg, gaze_x, gaze_y):
        """Run SAM segmentation given one RGB frame and gaze coordinate."""
        # Convert ROS Image to OpenCV
        image = self.bridge.imgmsg_to_cv2(image_msg, "rgb8")

        # Prepare gaze as a single point for SAM
        points = np.array([[gaze_x, gaze_y]])
        labels = np.array([1])

        self.get_logger().info(f"Running SAM on gaze point ({gaze_x:.1f}, {gaze_y:.1f})...")

        masks, scores = self.sam_client.sync_segment_request(image, points, labels)

        if masks is None or len(masks) == 0:
            self.get_logger().warn("No segmentation mask returned from SAM.")
            return

        # Select best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        score = scores[best_idx]

        self.get_logger().info(f"SAM segmentation done. Score = {score:.3f}")

        # Display results
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(points, labels, plt.gca())
        plt.title(f"SAM mask (score={score:.3f})", fontsize=18)
        plt.axis("off")

        save_path = os.path.join(
            os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
            "results/masks/aria_sam_masked.jpg"
        )
        plt.savefig(save_path)
        plt.show()
        plt.pause(1.0)

        # Crop image region based on mask
        y_mask, x_mask = np.where(np.squeeze(mask) == 1)
        if x_mask.size == 0 or y_mask.size == 0:
            self.get_logger().warn("Mask is empty, nothing to crop.")
            return

        x_min, x_max = x_mask.min(), x_mask.max()
        y_min, y_max = y_mask.min(), y_mask.max()
        cropped_img = image[y_min:y_max + 1, x_min:x_max + 1, :]

        self.get_logger().info(
            f"Cropped region: x[{x_min}:{x_max}], y[{y_min}:{y_max}], shape={cropped_img.shape}"
        )

        cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
        # Save cropped image
        cv2.imwrite(
            os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            "results/aria_sam_cropped.jpg"), cropped_rgb
        )
        self.get_logger().info("Saved cropped image to aria_sam_cropped.jpg")

        gaze_point = Point()
        gaze_point.x = float(gaze_x)
        gaze_point.y = float(gaze_y)
        self.gaze_pub.publish(gaze_point)

        cropped_msg = self.bridge.cv2_to_imgmsg(cropped_rgb, encoding="rgb8")
        self.image_pub.publish(cropped_msg)