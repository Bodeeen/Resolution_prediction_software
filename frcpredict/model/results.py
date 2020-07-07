from dataclasses import dataclass, field
from typing import Optional, Union, List

import numpy as np
from PySignal import Signal
from dataclasses_json import dataclass_json, config as json_config
from marshmallow import fields
from scipy.interpolate import interp1d

from frcpredict.util import dataclass_internal_attrs, observable_property
from .run_instance import RunInstance


@dataclass_json
@dataclass
class FrcCurve:
    range_values: List[float]

    x: np.ndarray = field(  # frequency
        metadata=json_config(
            encoder=lambda x: np.array(x).astype(float).tolist(),
            decoder=np.array,
            mm_field=fields.List
        ))

    y: np.ndarray = field(  # [0, 1]
        metadata=json_config(
            encoder=lambda x: np.array(x).astype(float).tolist(),
            decoder=np.array,
            mm_field=fields.List
        ))

    def resolution_at_threshold(self, threshold: float) -> Optional[float]:
        """
        Returns the resolution at a certain threshold, or None if the curve doesn't cross the
        threshold.
        """

        try:
            # Prevent issues when threshold line crosses two points by creating a copy of the y
            # value array and modifying it so that it doesn't happen
            y = np.copy(self.y)
            for i in range(1, len(y)):
                if y[i] > y[i - 1]:
                    y[i] = y[i - 1]

            return 1 / interp1d(y, self.x)(threshold)
        except ValueError:
            # Probably raised because the entered threshold is outside the interpolation range
            return None

@dataclass_json
@dataclass
class FrcSimulationResults:
    run_instance: RunInstance

    range_paths: List[List[Union[int, str]]]

    frc_curves: np.ndarray = field(  # array of FrcCurve
        metadata=json_config(
            encoder=lambda x: np.array(x).tolist(),
            decoder=lambda x: np.frompyfunc(FrcCurve.from_dict, 1, 1)(np.array(x)),
            mm_field=fields.List
        ))


@dataclass_json
@dataclass_internal_attrs(
    results_changed=Signal, range_value_index_changed=Signal, threshold_changed=Signal
)
@dataclass
class FrcSimulationResultsView:
    results: Optional[FrcSimulationResults] = observable_property(
        "_results", default=None, signal_name="results_changed", emit_arg_name="results"
    )

    range_value_indices: List[float] = observable_property(
        "_range_value_indices", default=[], signal_name="range_value_index_changed",
        emit_arg_name="range_value_indices"
    )

    threshold: float = observable_property(
        "_threshold", default=0.15, signal_name="threshold_changed", emit_arg_name="threshold"
    )

    def set_range_value(self, index_of_range: int, index_in_range: int) -> None:
        """
        Sets which index in the list of possible values (index_in_range) should be used when
        displaying the curve. index_of_range gives the index of the ranged value to adjust.
        """
        self.range_value_indices[index_of_range] = index_in_range
        self.range_value_index_changed.emit(self.range_value_indices)
