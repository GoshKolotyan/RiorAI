from typing import Dict, List


from src.utils.logger import setup_loger
from src.core.configs import Configs, ConfigsLoader
from src.floor_plane.floor_plane_preprocessing import (
    ObjectCleaner,
    OverlappingChecker,
    count_elements_in_bathroom,
)
from src.floor_plane.area_calulation_logic import NearestDoorFinder, BathroomMetricsAnalyzer
from src.floor_plane.floor_plane_model_loader import YOLOModel


def main(image_data: str, door_size_in_m: float, wall_height: float, configs: Configs) -> List[Dict]:
    logger = setup_loger(name="floor_plane_logger")

    BATHROOM_INFO = []
    
    try:
        logger.info("Step 1: Loading YOLO model and processing image")
        coordinates_dict, original_image = YOLOModel(
                image_path=image_data,
                weights=configs.floor_plane_detection.model.weights
            ).__call__()

        # Step 2: Object Cleaning
        # logger.info("Step 2: Cleaning objects")
        cleaner = ObjectCleaner(original_image)
        cleaned_image = cleaner(yolo_boxes=coordinates_dict)

        # Step 3: Overlapping check
        # logger.info("Step 3: Checking for overlapping bathrooms")
        filtered_bathroom_boxes = OverlappingChecker(cleaned_image, coordinates_dict).filter_overlapping_bathrooms()
        # logger.info(f"Filtered bathroom boxes: {len(filtered_bathroom_boxes)} found")

        # Step 4: Find nearest doors
        # logger.info("Step 4: Finding nearest doors")
        nearest_doors = NearestDoorFinder(coordinates_dict).find_nearest_door()
        # logger.info(f"Nearest doors found: {len(nearest_doors)}")

        # Step 5: Analyze each bathroom
        # logger.info("Step 5: Analyzing bathrooms")
        for idx, bathroom_bbox in enumerate(filtered_bathroom_boxes):
            try:
                bathroom = {}
                # logger.info(f"Processing bathroom {idx + 1}")

                nearest_door_info = nearest_doors.get(f"Bathroom_{idx}", None)
                if nearest_door_info is None or nearest_door_info.get("door_coords") is None:
                    # logger.warning(f"No door found for bathroom {idx + 1}, skipping")
                    continue

                door_bbox = nearest_door_info["door_coords"]

                # Analyze bathroom metrics
                (perimeter_meters, area_meters, wall_area, countures) = BathroomMetricsAnalyzer(
                    image=cleaned_image,
                    bathroom_bbox=bathroom_bbox,
                    door_bbox=door_bbox,
                    wall_height=wall_height,#configs.floor_plane_detection.constants.wall_height,
                    door_size_in_m=door_size_in_m#configs.floor_plane_detection.constants.door_size_in_meters
                ).visualize_contours_and_line()

                # Count elements in bathroom
                elements = count_elements_in_bathroom(
                    bathroom_bbox,
                    coordinates_dict,
                    cleaned_image.shape[0],
                    cleaned_image.shape[1],
                )

                # Build bathroom info dictionary
                bathroom["id"] = idx + 1
                bathroom["Area"] = area_meters
                bathroom["Polygone"] = countures
                bathroom["Perimeter"] = perimeter_meters
                bathroom["Wall_Area"] = wall_area
                bathroom["Elements"] = elements
                BATHROOM_INFO.append(bathroom)
                # logger.info(f"Successfully processed bathroom {idx + 1}")

            except Exception as e:
                # logger.error(f"Error processing bathroom {idx + 1}: {str(e)}")
                continue

        # logger.info(f"Processing complete. Found {len(BATHROOM_INFO)} valid bathrooms")
        return BATHROOM_INFO

    except Exception as e:
        # logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    from PIL import Image
    from pprint import pprint
    configs = ConfigsLoader().load_configs()
    img_path = "data/samples/floor_plane_samples/image_1.jpg"
    image = Image.open(img_path).convert("RGB")
    # pprint(configs)
    res = main(image_data=img_path, door_size_in_m=.8, wall_height=2.8, configs=configs)

