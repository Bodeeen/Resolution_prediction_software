from dataclasses import dataclass
from typing import Optional, List

import numpy as np
from PySignal import Signal

from frcpredict.model import SimulationResults
from frcpredict.util import dataclass_internal_attrs, dataclass_with_properties, observable_property


@dataclass
class SampleImage:
    """
    A description of a loaded sample image.
    """

    id: Optional[str] = None

    imageArr: np.ndarray = np.zeros((1, 1))


@dataclass_with_properties
@dataclass_internal_attrs(
    resultsChanged=Signal, inspectedMultivalueIndexChanged=Signal,
    multivalueValueIndexChanged=Signal, sampleImageChanged=Signal, thresholdChanged=Signal
)
@dataclass
class SimulationResultsView:
    """
    A representation of a view of results of a full simulation run.
    """

    results: Optional[SimulationResults] = observable_property(
        "_results", default=None, signal_name="resultsChanged", emit_arg_name="results"
    )

    inspectedMultivalueIndex: int = observable_property(  # -1 if no multivalue inspected
        "_inspectedMultivalueIndex", default=-1, signal_name="inspectedMultivalueIndexChanged",
        emit_arg_name="inspectedMultivalueIndex"
    )

    multivalueValueIndices: List[int] = observable_property(  # viewed multivalue value indices
        "_multivalueValueIndices", default=list, signal_name="multivalueValueIndexChanged",
        emit_arg_name="multivalueValueIndices"
    )

    sampleImage: Optional[SampleImage] = observable_property(  # loaded sample image
        "_sampleImage", default=None, signal_name="sampleImageChanged", emit_arg_name="sampleImage"
    )

    threshold: float = observable_property(
        "_threshold", default=0.15, signal_name="thresholdChanged", emit_arg_name="threshold"
    )

    def setMultivalueValue(self, indexOfMultivalue: int, indexInMultivalue: int) -> None:
        """
        Sets which index in the list of possible values (indexInMultivalue) should be used when
        displaying the results. indexOfMultivalue gives the index of the multivalue to adjust.
        """
        self.multivalueValueIndices[indexOfMultivalue] = indexInMultivalue
        self.multivalueValueIndexChanged.emit(self.multivalueValueIndices)
