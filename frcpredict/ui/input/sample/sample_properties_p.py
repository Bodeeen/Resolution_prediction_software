from typing import Union

from PyQt5.QtCore import pyqtSlot

from frcpredict.model import Multivalue, SampleProperties
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import connectMulti
from .sample_structure_picker_dialog_w import SampleStructurePickerDialog


class SamplePropertiesPresenter(BasePresenter[SampleProperties]):
    """
    Presenter for the sample properties widget.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: SampleProperties) -> None:
        # Disconnect old model event handling
        try:
            self._model.basic_field_changed.disconnect(self._onBasicFieldChange)
            self._model.loaded_structure_id_changed.disconnect(self._onPresetNameChange)
        except AttributeError:
            pass

        # Set model
        self._model = model

        # Trigger model change event handlers
        self._onBasicFieldChange(model)
        self._onPresetNameChange(model.loaded_structure_id)

        # Prepare model events
        model.basic_field_changed.connect(self._onBasicFieldChange)
        model.loaded_structure_id_changed.connect(self._onPresetNameChange)

    # Methods
    def __init__(self, widget) -> None:
        super().__init__(SampleProperties(), widget)

        # Prepare UI events
        connectMulti(widget.spectralPowerChanged, [float, Multivalue],
                     self._uiSpectralPowerChange)
        connectMulti(widget.labellingDensityChanged, [float, Multivalue],
                     self._uiLabellingDensityChange)
        connectMulti(widget.KOriginChanged, [float, Multivalue],
                     self._uiKOriginChange)

        widget.loadSampleStructureClicked.connect(self._uiClickLoadSampleStructure)
        widget.unloadSampleStructureClicked.connect(self._uiClickUnloadSampleStructure)

    # Model event handling
    def _onBasicFieldChange(self, model: SampleProperties) -> None:
        """ Loads basic model fields (e.g. ints) into the widget. """
        self.widget.updateBasicFields(model)

    def _onPresetNameChange(self, presetName: str) -> None:
        """ Loads basic model fields (e.g. ints) into the widget. """
        self.widget.updatePresetLoaded(presetName is not None)

    # UI event handling
    @pyqtSlot(float)
    @pyqtSlot(Multivalue)
    def _uiSpectralPowerChange(self, value: Union[float, Multivalue[float]]) -> None:
        self.model.spectral_power = value

    @pyqtSlot(float)
    @pyqtSlot(Multivalue)
    def _uiLabellingDensityChange(self, value: Union[float, Multivalue[float]]) -> None:
        self.model.labelling_density = value

    @pyqtSlot(float)
    @pyqtSlot(Multivalue)
    def _uiKOriginChange(self, value: Union[float, Multivalue[float]]) -> None:
        self.model.K_origin = value

    @pyqtSlot()
    def _uiClickLoadSampleStructure(self) -> None:
        sampleStructure, okClicked = SampleStructurePickerDialog.getSampleStructure(self.widget)

        if okClicked:
            self.model.load_structure(sampleStructure)

    @pyqtSlot()
    def _uiClickUnloadSampleStructure(self) -> None:
        self.model.loaded_structure_id = None
