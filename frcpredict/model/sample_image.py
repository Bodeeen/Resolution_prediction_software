from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SampleImage:
    """
    A description of a sample image.
    """

    id: Optional[str] = None

    image_arr: np.ndarray = np.zeros((1, 1))
