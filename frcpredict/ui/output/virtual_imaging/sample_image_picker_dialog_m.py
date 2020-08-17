from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PySignal import Signal

from frcpredict.model import Array2DPatternData
from frcpredict.util import dataclass_with_properties, dataclass_internal_attrs, observable_property


@dataclass_with_properties
@dataclass_internal_attrs(fluorophoresPerUnitChanged=Signal)
@dataclass
class SampleImagePickerModel:
    """
    Model for the sample image picker widget.
    """

    image: Array2DPatternData = field(default_factory=Array2DPatternData)

    sampleStructureId: Optional[str] = None

    fluorophoresPerUnit: float = observable_property(
        "_fluorophoresPerUnit", default=1.0, signal_name="fluorophoresPerUnitChanged",
        emit_arg_name="fluorophoresPerUnit"
    )

    fromFile: bool = False

    def getImageWithScaledValues(self) -> np.ndarray:
        """
        Returns the sample image with the values scaled in accordance with the value of the
        fluorophroes per unit field.
        """
        return self.image.get_numpy_array() * self.fluorophoresPerUnit
