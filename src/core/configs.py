import yaml
from pathlib import Path
from src.core.models import Configs

class ConfigsLoader:
    def __init__(self, yaml_path="configs/configs.yaml"):
        self.yaml_path = yaml_path
        # Find project root by looking for common project files
        current_dir = Path(__file__).parent
        project_root = self._find_project_root(current_dir)
        self.path = project_root / yaml_path

    def _find_project_root(self, start_path: Path) -> Path:
        """Find project root by looking for common project indicators"""
        indicators = ['pyproject.toml', 'setup.py', '.git', 'requirements.txt']
        current = start_path

        while current != current.parent:
            if any((current / indicator).exists() for indicator in indicators):
                return current
            current = current.parent

        # Fallback to current working directory
        return Path.cwd()

    def load_configs(self) -> Configs:
        """Load and return validated Configs object"""
        with open(self.path, 'r') as file:
            config_data = yaml.safe_load(file)
        return Configs.from_dict(config_data)

    def floor_plane_configs(self):
        """Legacy method for backward compatibility"""
        with open(self.path, 'r') as file:
            config_data = yaml.safe_load(file)
        return config_data.get('FLOOR_PLANE_DETECTION', {})


# if __name__ == "__main__":
#     from pprint import pprint
#     loader = ConfigsLoader()
#     res = loader.load_configs()
#     pprint(res.model_dump())