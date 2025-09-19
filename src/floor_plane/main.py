from typing import Dict, List
from .floor_plane_preprocessing import (
    ObjectCleaner,
    OverlappingChecker,
    count_elements_in_bathroom,
)
from .area_calulation_logic import NearestDoorFinder, BathroomMetricsAnalyzer
from .floor_plane_model_loader import YOLOModel



def main(image_data:str, door_size_in_m:float, wall_height:float ) -> List[Dict]:
    """
    Main function to process the image and extract bathroom information.
    Memory usage at each step is logged and saved into a CSV file.
    """
    print("Main function started")

    BATHROOM_INFO = []
    
    # Step 1: Process image and get path
    coordinates_dict, original_image = YOLOModel(image_data).__call__()

        
    # Step 2: Object Cleaning
    cleaner = ObjectCleaner(original_image)
    cleaned_image = cleaner(yolo_boxes=coordinates_dict)
    
    # Step 3: Overlapping check
    filtered_bathroom_boxes = OverlappingChecker(cleaned_image, coordinates_dict).filter_overlapping_bathrooms()

    # # Step 4: Find nearest doors
    nearest_doors = NearestDoorFinder(coordinates_dict).find_nearest_door()
    

    # Step 5: Analyze each bathroom
    for idx, bathroom_bbox in enumerate(filtered_bathroom_boxes):
        bathroom = {}

        nearest_door_info = nearest_doors.get(f"Bathroom_{idx}", None)
        if nearest_door_info is None or nearest_door_info["door_coords"] is None:
            continue

        door_bbox = nearest_door_info["door_coords"]

        (perimeter_meters, area_meters, wall_area, countures) = BathroomMetricsAnalyzer(
            image=cleaned_image, bathroom_bbox=bathroom_bbox, door_bbox=door_bbox, wall_height = wall_height,door_size_in_m=door_size_in_m
        ).visualize_contours_and_line()
    
        elements = count_elements_in_bathroom(
            bathroom_bbox,
            coordinates_dict,
            cleaned_image.shape[0],
            cleaned_image.shape[1],
        )

        bathroom["id"] = idx + 1
        bathroom["Area"] = area_meters
        bathroom["Polygone"] = countures
        bathroom["Perimeter"] = perimeter_meters
        bathroom["Wall_Area"] = wall_area
        bathroom["Elements"] = elements
        BATHROOM_INFO.append(bathroom)

    
    return BATHROOM_INFO

# if __name__ == "__main__":
#     from PIL import Image
#     img_path = "../test/floor_plane_test/Sample-floor-plan-image-with-the-specification-of-different-room-sizes-and-furniture.jpg"
#     image = Image.open(img_path).convert("RGB")
#     res = main(image_data=image)
#     pprint(res)

