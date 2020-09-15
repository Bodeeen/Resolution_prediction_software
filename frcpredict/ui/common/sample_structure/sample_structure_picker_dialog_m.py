from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from PySignal import Signal

from frcpredict.model import DisplayableSample, SampleStructure
from frcpredict.util import dataclass_with_properties, dataclass_internal_attrs, observable_property


class SampleType(Enum):
    """
    All supported sample types.
    """
    fromInput = "fromInput"
    random = "random"
    pairs = "pairs"
    lines = "lines"
    file = "file"


@dataclass_with_properties
@dataclass_internal_attrs(displayableSampleChanged=Signal)
@dataclass
class SampleStructurePickerModel:
    """
    Model for the sample picker dialog.
    """

    inputLoadedSample: Optional[DisplayableSample] = None

    _displayableSample: DisplayableSample = field(default_factory=SampleStructure)

    _displayableSampleType: Optional[SampleType] = None

    def isSampleLoaded(self) -> bool:
        """ Returns whether any sample has been loaded. """
        return self._displayableSample.is_loaded()

    def getLoadedSample(self) -> Tuple[DisplayableSample, SampleType]:
        """ Returns the currently loaded sample. """
        return self._displayableSample, self._displayableSampleType

    def loadSample(self, displayableSample: DisplayableSample,
                   displayableSampleType: SampleType) -> None:
        """ Loads a sample. """
        self._displayableSample = displayableSample
        self._displayableSampleType = displayableSampleType
        self.displayableSampleChanged.emit(self._displayableSample, self._displayableSampleType)

    def loadInputSample(self) -> None:
        """ Loads the sample set as inputLoadedSample. """
        self.loadSample(self.inputLoadedSample, SampleType.fromInput)
