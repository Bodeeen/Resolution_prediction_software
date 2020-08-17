from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
from PySignal import Signal

from frcpredict.util import dataclass_with_properties, dataclass_internal_attrs, observable_property
from ..controls.output_director_m import ViewOptions


class Plot(Enum):
    frc = "frc"
    inspection = "inspection"


@dataclass_with_properties
@dataclass_internal_attrs(
    currentPlotChanged=Signal, crosshairToggled=Signal,
    frcCurveChanged=Signal, viewOptionsChanged=Signal
)
@dataclass
class FrcResultsModel:
    """
    Model for the FRC/resolution results widget.
    """

    currentPlot: Plot = observable_property(
        "_currentPlot", default=Plot.frc, signal_name="currentPlotChanged",
        emit_arg_name="currentPlot"
    )

    crosshairEnabled: bool = observable_property(
        "_crosshairEnabled", default=True, signal_name="crosshairToggled",
        emit_arg_name="crosshairEnabled"
    )

    frcCurve: Optional[Tuple[np.ndarray, np.ndarray]] = observable_property(
        "_frcCurve", default=None, signal_name="frcCurveChanged",
        emit_arg_name="frcCurve"
    )

    viewOptions: ViewOptions = observable_property(
        "_viewOptions", default=ViewOptions(), signal_name="viewOptionsChanged",
        emit_arg_name="viewOptions"
    )
