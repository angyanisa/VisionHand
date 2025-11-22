#!/usr/bin/env python3
"""
clip_service_node.py
ROS2 service that runs CLIP zero-shot classification on a received image.

Service name: /clip/classify
Service type: custom (see below)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from example_interfaces.srv import GetParameters  # we'll make a simple custom-like service
from cv_bridge import CvBridge
import torch
import clip
import numpy as np
from std_msgs.msg import String

# You can define your own service message in ROS2 (recommended),
# e.g., `srv/ClipClassify.srv`:
# ---
# sensor_msgs/Image image
# ---
# std_msgs/String label
# float32 score
# but here we'll simulate the request/response using an Image subscription + String pub for simplicity.

class CLIPClassifierNode(Node):
    def __init__(self):
        super().__init__("clip_classifier_node")

        self.bridge = CvBridge()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # candidate labels
        self.labels = [
            "person", "chair", "table", "mug", "bottle", "phone",
            "keyboard", "dog", "cat", "book", "laptop", "monitor"
        ]
        self.text_tokens = clip.tokenize(self.labels).to(self.device)

        self.image_sub = self.create_subscription(Image, "/sam/cropped_image", self.image_cb, 10)
        self.result_pub = self.create_publisher(String, "/clip/label", 10)

        self.get_logger().info("CLIPClassifierNode ready: listening to /sam/cropped_image")

    def image_cb(self, msg: Image):
        """When a cropped image arrives, classify it."""
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        label, score = self.run_clip(image)
        out = String()
        out.data = f"{label}|{score:.4f}"
        self.result_pub.publish(out)
        self.get_logger().info(f"CLIP label: {label} (score={score:.3f})")

    def run_clip(self, img_rgb: np.ndarray):
        """Run CLIP on a numpy RGB image."""
        import PIL.Image
        pil = PIL.Image.fromarray(img_rgb)
        image_input = self.preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features = self.model.encode_text(self.text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs_np = probs.cpu().numpy().squeeze()
        best_idx = int(np.argmax(probs_np))
        return self.labels[best_idx], float(probs_np[best_idx])


def main(args=None):
    rclpy.init(args=args)
    node = CLIPClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
