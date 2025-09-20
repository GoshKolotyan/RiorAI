from typing import Dict, Optional
from pydantic import BaseModel, Field



class FloorPlaneCategoryHashMap(BaseModel):
    """Mapping of room/object categories to integer IDs"""
    bath: int = Field(alias="Bath", description="Bath category ID")
    bathroom: int = Field(alias="Bathroom", description="Bathroom category ID")
    door: int = Field(alias="Door", description="Door category ID")
    shower_cabin: int = Field(alias="Shower Cabin", description="Shower cabin category ID")
    sink: int = Field(alias="Sink", description="Sink category ID")
    toilet: int = Field(alias="Toilet", description="Toilet category ID")
    wash_machine: int = Field(alias="Wash Machine", description="Washing machine category ID")
    
    class Config:
        validate_by_name = True
        populate_by_name = True

    def get_category_name(self, category_id: int) -> Optional[str]:
        """Get category name by ID"""
        id_to_name = {
            self.bath: "Bath",
            self.bathroom: "Bathroom", 
            self.door: "Door",
            self.shower_cabin: "Shower Cabin",
            self.sink: "Sink",
            self.toilet: "Toilet",
            self.wash_machine: "Wash Machine",
        }
        return id_to_name.get(category_id)

    def get_category_id(self, category_name: str) -> Optional[int]:
        """Get category ID by name"""
        name_to_id = {
            "Bath": self.bath,
            "Bathroom": self.bathroom,
            "Door": self.door,
            "Shower Cabin": self.shower_cabin,
            "Sink": self.sink,
            "Toilet": self.toilet,
            "Wash Machine": self.wash_machine,
        }
        return name_to_id.get(category_name)


class FloorPlaneConstants(BaseModel):
    """Application constants for floor plan processing"""
    door_size_in_meters: float = Field(
        alias="DOOR_SIZE_IN_METERS", 
        gt=0, 
        description="Standard door size in meters"
    )
    hashmap: FloorPlaneCategoryHashMap = Field(
        alias="HASHMAP", 
        description="Category to ID mapping"
    )
    scale_constant: float = Field(
        alias="SCALE_CONSTANT", 
        gt=0, 
        description="Scale factor for measurements"
    )
    wall_height: float = Field(
        alias="WALL_HEIGHT", 
        gt=0, 
        description="Standard wall height in meters"
    )
    
    class Config:
        validate_by_name = True


class ModelConfig(BaseModel):
    """Model configuration settings"""
    weights: str = Field(
        alias="WEIGHTS", 
        description="Path to model weights file"
    )
    class Config:
        validate_by_name = True


class FloorPlaneModelConfig(BaseModel):  # Fixed typo here
    """Floor plane model configuration"""
    constants: FloorPlaneConstants = Field(
        alias="CONSTANTS", 
        description="Floor plan constants"
    )
    model: ModelConfig = Field(
        alias="MODEL", 
        description="Model configuration"
    )
    
    class Config:
        validate_by_name = True


class Configs(BaseModel):
    """Main configuration container"""
    floor_plane_detection: FloorPlaneModelConfig = Field(
        alias="FLOOR_PLANE_DETECTION", 
        description="Floor plane detection configuration"
    )
    
    class Config:
        validate_by_name = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "Configs":
        """Create config from dictionary"""
        return cls(**config_dict)

    def get_floor_plane_config(self) -> FloorPlaneModelConfig:
        """Quick access to floor plane config"""
        return self.floor_plane_detection



class FloorPlaneInput(BaseModel):
    # constants:dict[str, any]
    pass 


class FloorPlaneOutput(BaseModel):
    pass 

class RenderedImageOutput(BaseModel):
    pass

class PilineOutput(BaseModel):
    pass