from PyQt5.QtCore import pyqtSignal

from frcpredict.model import Pulse, PulseType, Pattern, PatternType
from frcpredict.ui import BaseWidget
from .pulse_properties_p import PulsePropertiesPresenter


class PulsePropertiesWidget(BaseWidget):
    """
    A widget where the user may set propreties of a specific pulse.
    """

    # Signals
    onTypeSelected = pyqtSignal()
    offTypeSelected = pyqtSignal()
    readoutTypeSelected = pyqtSignal()

    wavelengthChanged = pyqtSignal(int)
    durationChanged = pyqtSignal(float)
    durationChangedByUser = pyqtSignal(float)
    maxIntensityChanged = pyqtSignal(float)
    illuminationPatternChanged = pyqtSignal(Pattern)

    moveLeftClicked = pyqtSignal()
    moveRightClicked = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

        self.editIlluminationPattern.setFieldName("Illumination Pattern")
        self.editIlluminationPattern.setAllowEditGenerationAmplitude(False)
        self.editIlluminationPattern.setAvailableGenerationTypes(
            [PatternType.gaussian, PatternType.doughnut, PatternType.airy]
        )
        
        # Connect forwarded signals
        self.radioTypeOn.clicked.connect(self.onTypeSelected)
        self.radioTypeOff.clicked.connect(self.offTypeSelected)
        self.radioTypeReadout.clicked.connect(self.readoutTypeSelected)

        self.editWavelength.valueChanged.connect(self.wavelengthChanged)
        self.editDuration.valueChanged.connect(self.durationChanged)
        self.editDuration.valueChangedByUser.connect(self.durationChangedByUser)
        self.editMaxIntensity.valueChanged.connect(self.maxIntensityChanged)
        self.editIlluminationPattern.valueChanged.connect(self.illuminationPatternChanged)

        self.btnMoveLeft.clicked.connect(self.moveLeftClicked)
        self.btnMoveRight.clicked.connect(self.moveRightClicked)

        # Initialize presenter
        self._presenter = PulsePropertiesPresenter(self)

    def setEditWavelengthEnabled(self, enabled: bool) -> None:
        """ Sets whether the field for editing the wavelength is enabled. """
        self.editWavelength.setEnabled(enabled)
    
    def setChangeOrderVisible(self, visible: bool) -> None:
        """
        Sets whether the buttons for changing the pulse's position in the scheme are visible.
        """
        # TODO: Fix these still taking up space when hidden
        self.lblOrder.setVisible(visible)
        self.btnMoveLeft.setVisible(visible)
        self.btnMoveRight.setVisible(visible)

    def value(self) -> Pulse:
        return self._presenter.model

    def setValue(self, model: Pulse) -> None:
        self._presenter.model = model

    def updateIlluminationPattern(self, pattern: Pattern) -> None:
        self.editIlluminationPattern.setValue(pattern, emitSignal=False)

    def updateBasicFields(self, model: Pulse) -> None:
        if model.pulse_type == PulseType.on:
            self.radioTypeOn.setChecked(True)
        elif model.pulse_type == PulseType.off:
            self.radioTypeOff.setChecked(True)
        elif model.pulse_type == PulseType.readout:
            self.radioTypeReadout.setChecked(True)

        self.editWavelength.setValue(model.wavelength)
        self.editDuration.setValue(model.duration)
        self.editMaxIntensity.setValue(model.max_intensity)
