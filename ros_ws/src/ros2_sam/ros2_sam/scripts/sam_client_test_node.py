import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from ament_index_python import get_package_share_directory

from ros2_sam.sam_client import SAMClient
from ros2_sam.utils import show_box, show_mask, show_points


def main(args: List[str] = None) -> None:
    rclpy.init(args=args)

    sam_client = SAMClient(
        node_name="sam_client",
        service_name="sam_server/segment",
    )

    try:
        image = cv2.imread(
            os.path.join(get_package_share_directory("ros2_sam"), "data/test_bottle.jpg")
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        points = np.array([[2016, 1512]])
        labels = np.array([1])

        masks, scores = sam_client.sync_segment_request(
            image, points, labels
        )

        sorted_indices = np.argsort(scores)
        second_best_idx = sorted_indices[-2]
        second_best_mask = masks[second_best_idx]
        second_best_score = scores[second_best_idx]

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(second_best_mask, plt.gca())
        show_points(points, labels, plt.gca())

        plt.title(f"Score: {second_best_score:.3f}", fontsize=18)
        plt.axis("off")
        plt.show()

        y_mask, x_mask = np.where(np.squeeze(second_best_mask) == 1)
        if x_mask.size == 0 or y_mask.size == 0:
            print("Mask is empty")
        else:
            x_min, x_max = x_mask.min(), x_mask.max()
            y_min, y_max = y_mask.min(), y_mask.max()
        print(f"x_min = {x_min}, x_max = {x_max}, y_min = {y_min}, y_max = {y_max}")

        img_array = np.asarray(image)
        print(f"image shape = {img_array.shape}")
        cropped_img = img_array[y_min:y_max+1, x_min:x_max+1, :]
        print(f"cropped image shape = {cropped_img.shape}")
        print(f"path: {os.path.dirname(os.path.realpath(__file__))}")
        cv2.imwrite(
            os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "results/test_bottle_cropped.jpg"), cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
        )
        # cv2.imwrite(
        #     os.path.join(get_package_share_directory("ros2_sam"), "results/test_bottle_cropped.jpg"), cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
        # )
        # cv2.imshow("Cropped image", cropped_img)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
