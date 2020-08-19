from PyQt5.QtCore import pyqtSignal, pyqtSlot

from frcpredict.model import SampleProperties, Multivalue
from frcpredict.ui import BaseWidget
from frcpredict.ui.util import connectMulti, PresetFileDirs, UserFileDirs
from .sample_properties_p import SamplePropertiesPresenter


class SamplePropertiesWidget(BaseWidget):
    """
    A widget where the user may set sample properties.
    """

    # Signals
    valueChanged = pyqtSignal(SampleProperties)
    modifiedFlagSet = pyqtSignal()

    spectralPowerChanged = pyqtSignal([float], [Multivalue])
    labellingDensityChanged = pyqtSignal([float], [Multivalue])
    KOriginChanged = pyqtSignal([float], [Multivalue])

    loadSampleStructureClicked = pyqtSignal()
    unloadSampleStructureClicked = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

        # Prepare UI elements
        self.configPanel.setModelType(SampleProperties)
        self.configPanel.setPresetsDirectory(PresetFileDirs.SampleProperties)
        self.configPanel.setStartDirectory(UserFileDirs.SampleProperties)
        self.configPanel.setValueGetter(self.value)
        self.configPanel.setValueSetter(self.setValue)

        self.updatePresetLoaded(False)

        # Connect own signal slots
        self.modifiedFlagSet.connect(self._onModifiedFlagSet)

        # Connect forwarded signals
        connectMulti(self.editSpectralPower.valueChanged, [float, Multivalue],
                     self.spectralPowerChanged)
        connectMulti(self.editLabellingDensity.valueChanged, [float, Multivalue],
                     self.labellingDensityChanged)
        connectMulti(self.editKOrigin.valueChanged, [float, Multivalue],
                     self.KOriginChanged)
        self.configPanel.dataLoaded.connect(self.modifiedFlagSet)

        self.btnLoadSampleStructure.clicked.connect(self.loadSampleStructureClicked)
        self.btnUnloadSampleStructure.clicked.connect(self.unloadSampleStructureClicked)

        # Initialize presenter
        self._presenter = SamplePropertiesPresenter(self)

    def value(self) -> SampleProperties:
        return self._presenter.model

    def setValue(self, model: SampleProperties, emitSignal: bool = True) -> None:
        self.configPanel.setLoadedPath(None)
        self._presenter.model = model

        if emitSignal:
            self.valueChanged.emit(model)

    def updateBasicFields(self, model: SampleProperties) -> None:
        self.editSpectralPower.setValue(model.spectral_power)
        self.editLabellingDensity.setValue(model.labelling_density)
        self.editKOrigin.setValue(model.K_origin)

    def updatePresetLoaded(self, loaded: bool) -> None:
        self.editSpectralPower.setEnabled(not loaded)
        self.editKOrigin.setEnabled(not loaded)
        self.btnLoadSampleStructure.setVisible(not loaded)
        self.btnUnloadSampleStructure.setVisible(loaded)

    # Event handling
    @pyqtSlot()
    def _onModifiedFlagSet(self, *_args, **_kwargs) -> None:
        self.configPanel.setModifiedFlag()
