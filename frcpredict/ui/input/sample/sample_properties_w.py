from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMenu

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

    inputPowerChangedByUser = pyqtSignal([float], [Multivalue])
    DOriginChangedByUser = pyqtSignal([float], [Multivalue])

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
                                      self.btnLoadSampleStructure])

        # Connect own signal slots
        self.modifiedFlagSet.connect(self._onModifiedFlagSet)

        # Connect forwarded signals
        connectMulti(self.editInputPower.valueChangedByUser, [float, Multivalue],
                     self.inputPowerChangedByUser)
        connectMulti(self.editDOrigin.valueChangedByUser, [float, Multivalue],
                     self.DOriginChangedByUser)
        self.configPanel.dataLoaded.connect(self.modifiedFlagSet)

        self.btnLoadSampleStructure.clicked.connect(self.loadSampleStructureClicked)

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
        # Update input fields enabled state
        self.editInputPower.setEnabled(not loaded)
        self.editDOrigin.setEnabled(not loaded)
        self.editInputPower.setStaticText("Automatic" if loaded else None)
        self.editDOrigin.setStaticText("Automatic" if loaded else None)

        # Update unload action enabled state
        actionMenu = QMenu()
        unloadAction = actionMenu.addAction("Unload sample structure",
                                            self.unloadSampleStructureClicked)
        unloadAction.setEnabled(loaded)
        self.btnLoadSampleStructure.setMenu(actionMenu)

    # Internal methods
    def updateStructure(self) -> None:
        """ Updates which multivalue-related actions are available to the user. """


    # Event handling
    @pyqtSlot()
    def _onModifiedFlagSet(self, *_args, **_kwargs) -> None:
        self.configPanel.setModifiedFlag()
