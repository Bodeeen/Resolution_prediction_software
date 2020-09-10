from typing import Union

from PyQt5.QtCore import pyqtSlot

from frcpredict.model import Multivalue, SampleProperties, ExplicitSampleProperties
from frcpredict.ui import BasePresenter, SampleStructurePickerDialog
from frcpredict.ui.util import connectMulti, connectModelToSignal, disconnectModelFromSignal


class SamplePropertiesPresenter(BasePresenter[SampleProperties]):
    """
    Presenter for the sample properties widget.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: SampleProperties) -> None:
        # Disconnect old model event handling
        try:
            self._model.data_loaded.disconnect(self._onDataLoaded)
            self._model.basic_properties.basic_field_changed.disconnect(self._onBasicFieldChange)
            disconnectModelFromSignal(self.model, self._modifiedFlagSlotFunc)
        except AttributeError:
            pass

        # Set model
        self._model = model

        # Trigger model change event handlers
        self._onDataLoaded(model)
        self._onBasicFieldChange(model.basic_properties)

        # Prepare model events
        model.data_loaded.connect(self._onDataLoaded)
        model.basic_properties.basic_field_changed.connect(self._onBasicFieldChange)
        self._modifiedFlagSlotFunc = connectModelToSignal(self.model, self.widget.modifiedFlagSet)

    # Methods
    def __init__(self, widget) -> None:
        super().__init__(SampleProperties(), widget)

        # Prepare UI events
        connectMulti(widget.inputPowerChanged, [float, Multivalue],
                     self._uiInputPowerChange)
        connectMulti(widget.DOriginChanged, [float, Multivalue],
                     self._uiDOriginChange)

        widget.loadSampleStructureClicked.connect(self._uiClickLoadSampleStructure)
        widget.unloadSampleStructureClicked.connect(self._uiClickUnloadSampleStructure)

    # Model event handling
    def _onDataLoaded(self, model: SampleProperties) -> None:
        self.widget.updateStructureLoaded(model.structure is not None)

    def _onBasicFieldChange(self, basicProperties: ExplicitSampleProperties) -> None:
        """ Loads basic model fields (e.g. ints) into the widget. """
        self.widget.updateBasicFields(basicProperties)

    # UI event handling
    @pyqtSlot(float)
    @pyqtSlot(Multivalue)
    def _uiInputPowerChange(self, value: Union[float, Multivalue[float]]) -> None:
        if self.model.structure is None:
            self.model.input_power = value

    @pyqtSlot(float)
    @pyqtSlot(Multivalue)
    def _uiDOriginChange(self, value: Union[float, Multivalue[float]]) -> None:
        if self.model.structure is None:
            self.model.D_origin = value

    @pyqtSlot()
    def _uiClickLoadSampleStructure(self) -> None:
        sampleStructure, okClicked = SampleStructurePickerDialog.getSampleStructure(self.widget)

        if okClicked:
            self.model.structure = sampleStructure

    @pyqtSlot()
    def _uiClickUnloadSampleStructure(self) -> None:
        self.model.structure = None
