import torch
import logging
import numpy as np

from PIL import Image
from ultralytics import YOLO
from typing import List, Dict, Tuple, Union
from warnings import filterwarnings

from .configs import FloorPlaneConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

filterwarnings("ignore")


class YOLOModel(FloorPlaneConfig):
    def __init__(self, image_path: str, config_path=None):
        super().__init__(config_path)
        self.device = self.device if torch.cuda.is_available() else "cpu"
        self.model = YOLO(self.weights).to(self.device)
        self.image = image_path
        self._warmup()

    def _warmup(self):
        dummy = torch.zeros((1, 3, 640, 640)).to(self.device)
        self.model.predict(source=dummy, device=self.device, verbose=False)
        logger.info("Warmup inference completed.")

    def infer_and_process(self) -> Tuple[Dict[str, List[List[float]]], Union[Image.Image, None]]:
        try:
            results = self.model.predict(
                source=self.image,
                device=self.device,
                # save=True,
                # save_txt=True,
                augment=False,
                verbose=False
            )
            processed_results = self._process_results(results)
            
            # Get the original image from results and convert to PIL Image
            if results and hasattr(results[0], 'orig_img') and results[0].orig_img is not None:
                # Convert from BGR to RGB by swapping the channels
                bgr_image = results[0].orig_img
                rgb_image = bgr_image[..., ::-1]  # Reverse the color channels
                original_image = Image.fromarray(rgb_image)
            else:
                original_image = None
            
            return processed_results, original_image
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            if self.device == "cuda":
                logger.info("Falling back to CPU.")
                self.device = "cpu"
                return self.infer_and_process()
            return {}, None

    def _process_results(self, results) -> Dict[str, List[List[float]]]:
        if not results:
            return {}

        boxes = results[0].boxes
        class_names = results[0].names
        results_output = {class_name: [] for class_name in class_names.values()}

        for index in range(len(boxes)):
            cls_id = int(boxes.cls[index])
            cls_name = class_names.get(cls_id, f"Unknown-{cls_id}")
            box_xywhn = boxes.xywhn[index].tolist()
            results_output[cls_name].append(box_xywhn)

        # logger.info(f"Inference results: {results_output}")
        return results_output

    def __call__(self) -> Tuple[Dict[str, List[List[float]]], Union[Image.Image, None]]:

        return self.infer_and_process()
    
    def get_original_image(self) -> Union[Image.Image, None]:
        _, image = self.infer_and_process()
        return image
    
    def __del__(self):
        # Clean up resources
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'image'):
            del self.image
        
        # Handle CUDA cleanup safely
        try:
            if torch and torch.cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            # In case torch or cuda is already gone
            pass

# if __name__ == "__main__":
#     img_path = "../test/floor_plane_test/test.png"
#     model = YOLOModel(img_path)
#     results, original_image = model()
    
#     if original_image is not None:
#         # original_image.save("original.png")
#         original_image.show()  # This will display the image in correct colors