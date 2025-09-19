import os
import yaml

class FloorPlaneConfig:
    def __init__(self, config_path=None):
        self.config_path = "configs.yaml"
        self.config = self._load_config()

        self._load_model_parameters()
        self._load_constants()

    def _load_config(self):
        """
        Load and parse the YAML configuration file.
        Returns:
            dict: Parsed configuration data.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file {self.config_path} not found!")

        with open(self.config_path, "r") as file:
            return yaml.safe_load(file)

    def _load_model_parameters(self):
        """Load model parameters from configuration."""
        try:
            model_cfg = self.config["FLOOR_PLANE_DETECTION"]["MODEL"]
            params = self.config["FLOOR_PLANE_DETECTION"]["PARAMETERS"]

            self.weights = model_cfg["WEIGHTS"]
            self.confidence_threshold = params["CONFIDENCE_THRESHOLD"]
            self.iou_threshold = params["IOU_THRESHOLD"]
            self.device = params["DEVICE"]
            self.save_txt = params.get("SAVE_TXT", False)
            self.save_image = params.get("SAVE_IMAGE", False)
            self.project_name = params.get("PROJECT_NAME", "runs/detect")
            self.run_name = params.get("RUN_NAME", "predict")
        except KeyError as e:
            raise KeyError(f"Missing required configuration key: {e}")

    def _load_constants(self):
        """Load constants from configuration with defaults for missing values."""
        constants = self.config["FLOOR_PLANE_DETECTION"]["CONSTANTS"]

        self.hashmap = constants["HASHMAP"]
        self.scale_constant = constants["SCALE_CONSTANT"]
        
        # Handle missing constants with default values
        # These weren't in your original YAML but were referenced in your code
        self.door_size_in_meters = constants.get("DOOR_SIZE_IN_METERS", 0.85)  # Default value
        self.wall_height = constants.get("WALL_HEIGHT", 2.4)  # Default value

    def get_config_summary(self):
        """Return a summary of the loaded configuration."""
        return {
            "weights": self.weights,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "hashmap": self.hashmap,
            "scale_constant": self.scale_constant,
            "door_size_in_meters": self.door_size_in_meters,
            "wall_height": self.wall_height
        }
