from PyQt5.QtCore import pyqtSignal, pyqtSlot

from frcpredict.model import Multivalue, SampleProperties, ExplicitSampleProperties
from frcpredict.ui import BaseWidget
from frcpredict.ui.util import connectMulti, setTabOrderForChildren, PresetFileDirs, UserFileDirs
from .sample_properties_p import SamplePropertiesPresenter


class SamplePropertiesWidget(BaseWidget):
    """
    A widget where the user may set sample properties.
    """

    # Signals
    valueChanged = pyqtSignal(SampleProperties)
    modifiedFlagSet = pyqtSignal()

    inputPowerChanged = pyqtSignal([float], [Multivalue])
    DOriginChanged = pyqtSignal([float], [Multivalue])

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

        self.updateStructureLoaded(False)

        setTabOrderForChildren(self, [self.configPanel, self.editInputPower, self.editDOrigin,
                                      self.btnLoadSampleStructure, self.btnUnloadSampleStructure])

        # Connect own signal slots
        self.modifiedFlagSet.connect(self._onModifiedFlagSet)

        # Connect forwarded signals
        connectMulti(self.editInputPower.valueChanged, [float, Multivalue],
                     self.inputPowerChanged)
        connectMulti(self.editDOrigin.valueChanged, [float, Multivalue],
                     self.DOriginChanged)
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

    def updateBasicFields(self, basicProperties: ExplicitSampleProperties) -> None:
        self.editInputPower.setValue(basicProperties.input_power)
        self.editDOrigin.setValue(basicProperties.D_origin)

    def updateStructureLoaded(self, loaded: bool) -> None:
        self.editInputPower.setEnabled(not loaded)
        self.editDOrigin.setEnabled(not loaded)

        self.editInputPower.setStaticText("Automatic" if loaded else None)
        self.editDOrigin.setStaticText("Automatic" if loaded else None)

        self.btnLoadSampleStructure.setVisible(not loaded)
        self.btnUnloadSampleStructure.setVisible(loaded)

    # Event handling
    @pyqtSlot()
    def _onModifiedFlagSet(self, *_args, **_kwargs) -> None:
        self.configPanel.setModifiedFlag()
