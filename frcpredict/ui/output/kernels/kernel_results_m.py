from dataclasses import dataclass
from typing import Optional

import numpy as np
from PySignal import Signal

from frcpredict.model import KernelType
from frcpredict.util import dataclass_with_properties, dataclass_internal_attrs, observable_property


@dataclass_with_properties
@dataclass_internal_attrs(kernelTypeChanged=Signal, kernels2DChanged=Signal)
@dataclass
class KernelResultsModel:
    """
    Model for the kernel results widget.
    """

    kernelType: KernelType = observable_property(
        "_kernelType", default=KernelType.exp_kernel, signal_name="kernelTypeChanged",
        emit_arg_name="kernelType"
    )

    kernels2D: Optional[np.ndarray] = observable_property(
        "_kernels2D", default=None, signal_name="kernels2DChanged",
        emit_arg_name="kernels2D"
    )
