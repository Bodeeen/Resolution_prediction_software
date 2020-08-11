from dataclasses import dataclass, field
from typing import Optional

from frcpredict.model import Array2DPatternData


@dataclass
class SampleImagePickerModel:
    """
    Model for the sample image picker widget.
    """

    image: Array2DPatternData = field(default_factory=Array2DPatternData)

    sampleStructureId: Optional[str] = None

    fromFile: bool = False
