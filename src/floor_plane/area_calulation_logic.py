import cv2
import numpy as np
import matplotlib.pyplot as plt


from typing import List, Dict, Tuple
from .configs import FloorPlaneConfig

confisgs = FloorPlaneConfig()

class NearestDoorFinder:
    def __init__(self, coordinates_dict: Dict[str, List[List[float]]]):
        if not isinstance(coordinates_dict, dict):
            raise ValueError("coordinates_dict must be a dictionary.")
        self.coordinates_dict = coordinates_dict

    def calculate_distance(self, coords_1: List[float], coords_2: List[float]) -> float:
        """Calculate Euclidean distance between two points."""
        center_x_1, center_y_1 = coords_1[:2]
        center_x_2, center_y_2 = coords_2[:2]
        return np.sqrt((center_x_1 - center_x_2) ** 2 + (center_y_1 - center_y_2) ** 2)

    def calculate_bathroom_distances(self) -> Dict[str, List[Dict[str, float]]]:
        """Calculate distances from each bathroom to all other objects."""
        distances_by_bathroom = {}
        bathroom_coords = self.coordinates_dict.get("Bathroom", [])

        for i, bathroom in enumerate(bathroom_coords):
            distances = []
            for label, coords_list in self.coordinates_dict.items():
                if label == "Bathroom":
                    continue
                for coords in coords_list:
                    distances.append(
                        {
                            "label": label,
                            "distance": self.calculate_distance(bathroom, coords),
                            "door_coords": coords,
                            "bath_coords": bathroom,
                        }
                    )
            distances_by_bathroom[f"Bathroom_{i}"] = distances

        return distances_by_bathroom

    def find_nearest_door(self) -> Dict[str, Dict[str, float]]:
        """Find the nearest door for each bathroom."""
        bathroom_distances = self.calculate_bathroom_distances()
        nearest_doors = {}

        for bathroom, distances in bathroom_distances.items():
            door_distances = [d for d in distances if d["label"] == "Door"]
            if not door_distances:
                print("Noo Doors")
            if door_distances:
                nearest = min(door_distances, key=lambda x: x["distance"])
                nearest_doors[bathroom] = {
                    "nearest_door_distance": nearest["distance"],
                    "bathroom_coords": nearest["bath_coords"],
                    "door_coords": nearest["door_coords"],
                }
            else:
                nearest_doors[bathroom] = {
                    "nearest_door_distance": None,
                    "bathroom_coords": None,
                    "door_coords": None,
                }

        return nearest_doors

    def process(self) -> Dict[str, Dict[str, float]]:
        """Run the full workflow: calculate distances and find nearest doors."""
        return self.find_nearest_door()


class BathroomMetricsAnalyzer(FloorPlaneConfig):
    def __init__(self, image: np.ndarray, 
                 bathroom_bbox: Tuple, 
                 door_bbox: List[float], 
                 wall_height:float=2.8 ,
                 door_size_in_m:float=0.85):
        super().__init__()
        self.bathroom_bbox = bathroom_bbox
        self.door_bbox = door_bbox
        self.door_size_m = door_size_in_m
        self.image = image
        self.wall_height = wall_height
        self.img_height, self.img_width = self.image.shape[:2]

    def yolo_to_pixel_coords(
        self, bbox, ratio=0.09
    ):  # for using in crop_bath it should be 0.1
        """Convert YOLO format bounding box to pixel coordinates with an additional margin ratio."""
        x_center, y_center, width, height = bbox
        x_center *= self.img_width
        y_center *= self.img_height
        width *= self.img_width
        height *= self.img_height

        margin_x = int(width * ratio / 2)
        margin_y = int(height * ratio / 2)

        x_min = max(0, int(x_center - width / 2) - margin_x)
        y_min = max(0, int(y_center - height / 2) - margin_y)
        x_max = min(self.img_width, int(x_center + width / 2) + margin_x)
        y_max = min(self.img_height, int(y_center + height / 2) + margin_y)

        return (x_min, y_min, x_max, y_max)

    def crop_bathroom(self):
        """Crop the bathroom area from the image using the bounding box."""
        x_min, y_min, x_max, y_max = self.yolo_to_pixel_coords(
            self.bathroom_bbox, ratio=0.09
        )  # ratio (0.1 is best) of adding balck color into image

        width = x_max - x_min
        height = y_max - y_min

        cropped_bathroom_with_margin = np.zeros((height, width, 3), dtype=np.uint8)

        cropped_bathroom = self.image[y_min:y_max, x_min:x_max]

        cropped_bathroom_with_margin[
            0 : cropped_bathroom.shape[0], 0 : cropped_bathroom.shape[1]
        ] = cropped_bathroom

        return cropped_bathroom_with_margin, (
            x_min,
            y_min,
            x_max,
            y_max,
        )  # Return the bounding box for further use

    def remove_background(self, image):
        """Remove everything except the walls in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)    

        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        mask = np.zeros_like(image)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            print("No contours found!")
            return None

        contour_image = mask.copy()
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_image, [largest_contour], -1, (255, 255, 255), 2)

        return contour_image

    def visualize_contours_and_line(self):
        """Visualize contours, remove background, and calculate perimeter using YOLO bounding boxes."""
        cropped_bathroom, bathroom_coords = self.crop_bathroom()
        walls_only = self.remove_background(cropped_bathroom)

        door_x_min, door_y_min, door_x_max, door_y_max = self.yolo_to_pixel_coords(
            self.door_bbox
        )

        cropped_height, cropped_width = cropped_bathroom.shape[:2]

        scale_x = cropped_width / (bathroom_coords[2] - bathroom_coords[0])
        scale_y = cropped_height / (bathroom_coords[3] - bathroom_coords[1])

        normalized_door_x_min = int((door_x_min - bathroom_coords[0]) * scale_x)
        normalized_door_y_min = int((door_y_min - bathroom_coords[1]) * scale_y)
        normalized_door_x_max = int((door_x_max - bathroom_coords[0]) * scale_x)
        normalized_door_y_max = int((door_y_max - bathroom_coords[1]) * scale_y)

        door_pixels = normalized_door_x_max - normalized_door_x_min
        # print(f"Door width in pixels: {door_pixels} pixels")

        gray = cv2.cvtColor(walls_only, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            print("No contours found!")
            return None

        # Draw contours on the cropped image
        contour_image = walls_only.copy()
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_image, [largest_contour], -1, (255, 0, 0), 3)

        perimeter_pixels = cv2.arcLength(largest_contour, True)

        pixels_per_meter = door_pixels / self.door_size_m
        perimeter_meters = perimeter_pixels / pixels_per_meter

        wall_area = self.wall_height * perimeter_meters

        area_pixels = cv2.contourArea(largest_contour)
        area_meters = area_pixels / (pixels_per_meter**2)

        # Shift contours to original image coordinate space
        x_offset, y_offset = bathroom_coords[0], bathroom_coords[1]
        # larges_shifter_counter = largest_contour + [x_offset, y_offset]
        larges_shifter_counter = largest_contour + np.array([[x_offset, y_offset]])


        original_image_with_contours = self.image.copy()
        cv2.drawContours(
            original_image_with_contours, [larges_shifter_counter], -1, (255, 0, 0), 3
        )

        # plt.figure(figsize=(18, 6))

        # plt.subplot(1, 3, 1)
        # plt.title("Cropped Bathroom Image with Walls Only")
        # plt.imshow(walls_only)
        # plt.axis("off")

        # plt.subplot(1, 3, 2)
        # plt.title("Contours on Cropped Image")
        # plt.imshow(contour_image)
        # plt.axis("off")

        # plt.subplot(1, 3, 3)
        # plt.title("Contours on Original Image")
        # plt.imshow(original_image_with_contours)
        # plt.axis("off")

        # plt.tight_layout()
        # plt.show()

        return (
            perimeter_meters,
            area_meters,
            wall_area,
            np.squeeze(larges_shifter_counter).tolist(),
        )
    # def visualize_contours_and_line(self, epsilon_factor=0.1):
    #     """Visualize contours clearly using simplified approximations."""

    #     cropped_bathroom, bathroom_coords = self.crop_bathroom()
    #     walls_only = self.remove_background(cropped_bathroom)

    #     door_x_min, door_y_min, door_x_max, door_y_max = self.yolo_to_pixel_coords(self.door_bbox)

    #     cropped_height, cropped_width = cropped_bathroom.shape[:2]

    #     scale_x = cropped_width / (bathroom_coords[2] - bathroom_coords[0])
    #     scale_y = cropped_height / (bathroom_coords[3] - bathroom_coords[1])

    #     normalized_door_x_min = int((door_x_min - bathroom_coords[0]) * scale_x)
    #     normalized_door_y_min = int((door_y_min - bathroom_coords[1]) * scale_y)
    #     normalized_door_x_max = int((door_x_max - bathroom_coords[0]) * scale_x)
    #     normalized_door_y_max = int((door_y_max - bathroom_coords[1]) * scale_y)

    #     door_pixels = normalized_door_x_max - normalized_door_x_min

    #     gray = cv2.cvtColor(walls_only, cv2.COLOR_BGR2GRAY)
    #     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #     # Optional Morphology to clean the image
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #     thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    #     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     if not contours:
    #         print("No contours found!")
    #         return None

    #     largest_contour = max(contours, key=cv2.contourArea)

    #     # Contour approximation for cleaner visualization
    #     epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
    #     approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    #     # Draw simplified contour on cropped image
    #     contour_image = walls_only.copy()
    #     cv2.drawContours(contour_image, [approx_contour], -1, (0, 255, 0), 3)

    #     perimeter_pixels = cv2.arcLength(approx_contour, True)

    #     pixels_per_meter = door_pixels / self.door_size_m
    #     perimeter_meters = perimeter_pixels / pixels_per_meter
    #     wall_area = self.wall_height * perimeter_meters

    #     area_pixels = cv2.contourArea(approx_contour)
    #     area_meters = area_pixels / (pixels_per_meter ** 2)

    #     # Shift contours back to original image coordinates
    #     x_offset, y_offset = bathroom_coords[0], bathroom_coords[1]
    #     shifted_contour = approx_contour + np.array([[x_offset, y_offset]])

    #     original_image_with_contours = self.image.copy()
    #     cv2.drawContours(original_image_with_contours, [shifted_contour], -1, (0, 255, 0), 3)

    #     # Visualization
    #     fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    #     axes[0].imshow(cv2.cvtColor(walls_only, cv2.COLOR_BGR2RGB))
    #     axes[0].set_title("Walls Only (Cropped)")
    #     axes[0].axis('off')

    #     axes[1].imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    #     axes[1].set_title("Simplified Contours (Cropped)")
    #     axes[1].axis('off')

    #     axes[2].imshow(cv2.cvtColor(original_image_with_contours, cv2.COLOR_BGR2RGB))
    #     axes[2].set_title("Simplified Contours (Original Image)")
    #     axes[2].axis('off')

    #     plt.tight_layout()
    #     plt.show()

    #     return (
    #         perimeter_meters,
    #         area_meters,
    #         wall_area,
    #         np.squeeze(shifted_contour).tolist(),
    #     )

