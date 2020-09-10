from dataclasses import dataclass

from PySignal import Signal
from dataclasses_json import dataclass_json

from frcpredict.util import (
    dataclass_internal_attrs, dataclass_with_properties, observable_property
)


@dataclass_json
@dataclass_with_properties
@dataclass_internal_attrs(
    canvas_inner_radius_changed=Signal, num_kernel_detection_iterations_changed=Signal
)
@dataclass
class SimulationSettings:
    """
    Advanced simulation settings.
    """

    canvas_inner_radius: float = observable_property(  # nanometres
        "_canvas_inner_radius", default=500.0,
        signal_name="canvas_inner_radius_changed",
        emit_arg_name="canvas_inner_radius"
    )

    num_kernel_detection_iterations: int = observable_property(
        "_num_kernel_detection_iterations", default=500000,
        signal_name="num_kernel_detection_iterations_changed",
        emit_arg_name="num_kernel_detection_iterations"
    )
