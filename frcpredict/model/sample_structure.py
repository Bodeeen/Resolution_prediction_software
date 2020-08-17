from dataclasses import dataclass, field
from typing import Optional

from dataclasses_json import dataclass_json

from frcpredict.util import get_sample_structure_data_file_paths
from .pattern_data import Array2DPatternData


@dataclass_json
@dataclass
class SampleStructureProperties:
    """
    Properties of a sample structure.
    """

    spectral_power: float

    K_origin: float

    name: Optional[str] = None


@dataclass_json
@dataclass
class SampleStructure:
    """
    A description of a predefined sample structure containing an identifier, properties, and image.
    """

    id: str = "none"

    properties: SampleStructureProperties = field(
        default_factory=lambda: SampleStructureProperties(spectral_power=1.0, K_origin=1.0)
    )

    image: Array2DPatternData = field(default_factory=Array2DPatternData)

    @classmethod
    def from_sample_data_directory_name(cls, dir_name: str) -> "SampleStructure":
        """ Loads a sample structure from the data in the given directory. """

        properties_file_path, image_file_path = get_sample_structure_data_file_paths(dir_name)

        with open(properties_file_path, "r") as jsonFile:
            properties = SampleStructureProperties.from_json(jsonFile.read())

        return cls(id=dir_name, properties=properties,
                   image=Array2DPatternData.from_image_file(image_file_path))
