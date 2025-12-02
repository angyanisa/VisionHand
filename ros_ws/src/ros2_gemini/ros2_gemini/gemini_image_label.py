import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

class GeminiImageLabel(Node):
    def __init__(self):
        super().__init__('gemini_image_label')

        self.bridge = CvBridge()
        self.create_subscription(Image, "/sam/cropped_image", self.image_callback, 10)

        self.publisher = self.create_publisher(String, '/gemini/detected_object', 10)

        # Set up Gemini client (use environment variable for API key)
        load_dotenv('/home/belle/VisionHand/ros_ws/.env')
        api_key = os.getenv("API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.get_logger().info('Gemini Image Label Node Initialized. Waiting for images...')

    def image_callback(self, msg):
        self.get_logger().info('Received image, processing...')
        try:
            # Convert ROS Image message to OpenCV format (BGR8)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # Convert OpenCV image to JPEG bytes
            success, buffer = cv2.imencode('.jpg', cv_image)
            if not success:
                self.get_logger().error("Could not encode image to JPEG")
                return
            image_bytes = buffer.tobytes()

            # Define the prompt and the image part
            prompt = ("What is the object part at the red dot? Choose your response from these options only: "
                "Bottle cap, Bottle body, Mug body, Mug handle, Door handle, Scissor, Spray bottle, "
                "None of the above. Do not add anything else to your response.")
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg'
            )

            # Send to Gemini API
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[image_part, prompt]
            )
            
            response = response.text.strip()

            self.get_logger().info(f'Gemini Response: {response}')

            detected_object = String()
            detected_object.data =response
            self.publisher.publish(detected_object)
            self.get_logger().info(f'Published detected object!')

        except Exception as e:
            self.get_logger().error(f"Error processing image or calling Gemini API: {e}")

def main(args=None):
    rclpy.init(args=args)
    gemini_image_label = GeminiImageLabel()
    rclpy.spin(gemini_image_label)
    gemini_image_label.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()