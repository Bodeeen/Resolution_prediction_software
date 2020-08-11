from typing import Optional

from PyQt5.QtCore import pyqtSlot

from frcpredict.model import SampleStructure
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import getArrayPixmap
from frcpredict.util import get_sample_structure_data_dir_names


class SampleStructurePickerPresenter(BasePresenter[SampleStructure]):
    """
    Presenter for the sample structure picker dialog.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: SampleStructure) -> None:
        self._model = model
        self._onModelSet(model)

    # Methods
    def __init__(self, widget) -> None:
        super().__init__(SampleStructure(), widget)

        # Prepare UI events
        widget.valueChanged.connect(self._uiValueChange)

        # Load sample structures
        self._loadSampleStructures()

    # Internal methods
    def _loadSampleStructures(self) -> None:
        self.widget.setAvailableStructures([
            SampleStructure.from_sample_data_directory_name(sampleDirName)
            for sampleDirName in get_sample_structure_data_dir_names()
        ])

    # Model event handling
    def _onModelSet(self, model: SampleStructure) -> None:
        """ Updates the preview and property fields based on the sample structure. """

        self.widget.updateStructure(model.id, model.properties)

        self.widget.updatePreview(
            getArrayPixmap(model.image.get_numpy_array())
        )

    # UI event handling
    @pyqtSlot(object)
    def _uiValueChange(self, sample: Optional[SampleStructure] = None) -> None:
        """ Loads the selected sample structure into the model. """

        if sample is None:
            return

        self.model = sample
