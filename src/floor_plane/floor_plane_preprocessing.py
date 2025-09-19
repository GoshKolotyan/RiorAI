
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple



class ObjectCleaner:
    """
    A utility class to remove (paint over) detected objects in an image
    based on YOLO bounding boxes, except for objects of a specified label.
    """

    def __init__(
        self,
        image: Image.Image,
        color: Tuple[int, int, int] = (255, 255, 255),
        margin_ratio: float = 0.00,
    ) -> None:
        self.exempt_label = "Bathroom"
        self.color = color
        self.margin_ratio = margin_ratio
        self.image = image

    def yolo_to_pixel_coords(
        self, bbox: List[float], img_width: int, img_height: int
    ) -> Tuple[int, int, int, int]:
        x_center, y_center, width, height = bbox

        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        margin_x = int(width * self.margin_ratio / 2)
        margin_y = int(height * self.margin_ratio / 2)

        x_min = max(0, int(x_center - width / 2) - margin_x)
        y_min = max(0, int(y_center - height / 2) - margin_y)
        x_max = min(img_width, int(x_center + width / 2) + margin_x)
        y_max = min(img_height, int(y_center + height / 2) + margin_y)

        return x_min, y_min, x_max, y_max

    def clean_objects(
        self, yolo_boxes: Dict[str, List[List[float]]]
    ) -> np.ndarray:
        img_array = np.array(self.image)
        img_height, img_width = img_array.shape[:2]
        cleaned_image = img_array.copy()

        for label, bboxes in yolo_boxes.items():
            if label == self.exempt_label:
                continue

            for x_min, y_min, x_max, y_max in map(
                lambda bbox: self.yolo_to_pixel_coords(bbox, img_width, img_height),
                bboxes,
            ):
                cleaned_image[y_min:y_max, x_min:x_max] = self.color

        return cleaned_image

    def __call__(
        self,
        yolo_boxes: Dict[str, List[List[float]]],
    ) -> np.ndarray:
        return self.clean_objects(yolo_boxes)


class OverlappingChecker:
    def __init__(
                self, 
                image: Image.Image, 
                bathroom_boxes: List[Tuple[float, float, float, float]]
    ):
        self.image = np.array(image)
        self.bathroom_boxes = bathroom_boxes.get("Bathroom", [])
        self.img_height, self.img_width = image.shape[:2]

    def yolo_to_pixel_coords(
        self, box: Tuple[float, float, float, float]
    ) -> Tuple[int, int, int, int]:
        x_center, y_center, width, height = box
        x1 = int((x_center - width / 2) * self.img_width)
        y1 = int((y_center - height / 2) * self.img_height)
        x2 = int((x_center + width / 2) * self.img_width)
        y2 = int((y_center + height / 2) * self.img_height)
        return x1, y1, x2, y2

    def pixel_to_yolo_coords(
        self, box: Tuple[int, int, int, int]
    ) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = box
        x_center = ((x1 + x2) / 2) / self.img_width
        y_center = ((y1 + y2) / 2) / self.img_height
        width = (x2 - x1) / self.img_width
        height = (y2 - y1) / self.img_height
        return x_center, y_center, width, height

    def get_box_area(self, box: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def is_overlapping(
        self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]
    ) -> bool:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        overlap_x1 = max(x1_1, x1_2)
        overlap_y1 = max(y1_1, y1_2)
        overlap_x2 = min(x2_1, x2_2)
        overlap_y2 = min(y2_1, y2_2)

        return overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2

    def filter_overlapping_bathrooms(self) -> List[Tuple[float, float, float, float]]:
        pixel_boxes = [self.yolo_to_pixel_coords(box) for box in self.bathroom_boxes]
        filtered_boxes = []

        for i, box1 in enumerate(pixel_boxes):
            is_nested = False
            for j, box2 in enumerate(pixel_boxes):
                if i != j and self.is_overlapping(box1, box2):
                    if self.get_box_area(box1) < self.get_box_area(box2):
                        is_nested = True
                        break  # Skip this box as it is nested or smaller
            if not is_nested:
                filtered_boxes.append(box1)

        normalized_boxes = [self.pixel_to_yolo_coords(box) for box in filtered_boxes]
        return normalized_boxes


def count_elements_in_bathroom(
    bathroom_bbox: List[float],
    detections: Dict[str, List[List[float]]],
    image_height: int,
    image_width: int,
    ratio: float = 0.10,
) -> Dict[str, int]:

    bx_center, by_center, bwidth, bheight = bathroom_bbox
    bx_min = int((bx_center - bwidth / 2) * image_width)
    by_min = int((by_center - bheight / 2) * image_height)
    bx_max = int((bx_center + bwidth / 2) * image_width)
    by_max = int((by_center + bheight / 2) * image_height)

    # Ratio
    margin_x = int((bx_max - bx_min) * ratio)
    margin_y = int((by_max - by_min) * ratio)
    bx_min -= margin_x
    by_min -= margin_y
    bx_max += margin_x
    by_max += margin_y

    bx_min = max(bx_min, 0)
    by_min = max(by_min, 0)
    bx_max = min(bx_max, image_width)
    by_max = min(by_max, image_height)

    pixel_bathroom_bbox = [bx_min, by_min, bx_max, by_max]

    element_counts = {
        "Bath": 0,
        "Shower Cabin": 0,
        "Sink": 0,
        "Toilet": 0,
        "Washing Machine": 0,
    }

    for label, objects in detections.items():
        if label == "Bathroom" or label == "Door":
            continue
        for obj in objects:
            x_center, y_center, width, height = obj
            x_min = int((x_center - width / 2) * image_width)
            y_min = int((y_center - height / 2) * image_height)
            x_max = int((x_center + width / 2) * image_width)
            y_max = int((y_center + height / 2) * image_height)

            inside = (
                x_min >= pixel_bathroom_bbox[0]
                and y_min >= pixel_bathroom_bbox[1]
                and x_max <= pixel_bathroom_bbox[2]
                and y_max <= pixel_bathroom_bbox[3]
            )

            if inside:
                element_counts[label] += 1

    return element_counts
