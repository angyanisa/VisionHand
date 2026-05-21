# ============================================================
# Imports — libraries we need
# ============================================================
import os
import tempfile
import threading
import rclpy                              # ROS 2 Python client
from rclpy.node import Node
from std_msgs.msg import String           # The message type Gemini publishes

from gtts import gTTS                     # Text-to-speech (Google Translate)
import pygame                             # For playing the resulting audio


class GeminiToAudio(Node):
    """Speaks labels published on /gemini/detected_object through the
    system's default audio output (route Bluetooth headphones there at
    the OS level - nothing to configure in code)."""

    # ============================================================
    # Setup — runs once when the node starts
    # ============================================================
    def __init__(self):
        super().__init__('gemini_to_audio')

        # Track what we last spoke, so we don't repeat ourselves
        self._last_spoken = None
        self._lock = threading.Lock()

        # Initialize the audio playback engine
        pygame.mixer.init()

        # Listen for messages from the Gemini labeler node
        self.create_subscription(
            String, '/gemini/detected_object', self.label_cb, 10
        )
        self.get_logger().info('gemini_to_audio ready. Listening for labels...')

    # ============================================================
    # Callback — runs every time Gemini publishes a label
    # ============================================================
    def label_cb(self, msg: String):
        label = (msg.data or '').strip()
        if not label:
            return

        # If Gemini returned "None of the above", don't speak it
        if label.lower().startswith('none'):
            self.get_logger().info(f'Skipping uninformative label: "{label}"')
            return

        # Don't speak the same thing twice in a row
        with self._lock:
            if label == self._last_spoken:
                self.get_logger().info(f'Skipping repeat label: "{label}"')
                return
            self._last_spoken = label

        # Speak it on a separate thread so the ROS node stays responsive
        threading.Thread(target=self._speak, args=(label,), daemon=True).start()

    # ============================================================
    # Speak — generates and plays audio
    # ============================================================
    def _speak(self, text: str):
        path = None
        try:
            self.get_logger().info(f'Speaking: "{text}"')

            # Generate speech audio (sends text to Google TTS, returns MP3)
            tts = gTTS(text=text, lang='en')

            # Save the MP3 to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                path = f.name
            tts.save(path)

            # Play the MP3 through the system's default audio output
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(50)
            pygame.mixer.music.unload()

        except Exception as e:
            self.get_logger().error(f'TTS failed: {e}')
        finally:
            # Clean up the temp file
            if path and os.path.exists(path):
                os.unlink(path)


# ============================================================
# Standard ROS 2 entry point boilerplate
# ============================================================
def main(args=None):
    rclpy.init(args=args)
    node = GeminiToAudio()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.mixer.quit()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()