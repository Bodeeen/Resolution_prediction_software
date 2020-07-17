from PyQt5.QtCore import pyqtSignal

from frcpredict.model import Pulse, Pattern, PatternType, Multivalue
from frcpredict.ui import BaseWidget
from frcpredict.ui.util import setFormLayoutRowVisibility, connectMulti
from .pulse_properties_p import PulsePropertiesPresenter


class PulsePropertiesWidget(BaseWidget):
    """
    A widget where the user may set propreties of a specific pulse.
    """

    # Signals
    wavelengthChanged = pyqtSignal([int], [Multivalue])
    durationChanged = pyqtSignal([float], [Multivalue])
    durationChangedByUser = pyqtSignal([float], [Multivalue])
    maxIntensityChanged = pyqtSignal([float], [Multivalue])
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
        connectMulti(self.editWavelength.valueChanged, [int, Multivalue],
                     self.wavelengthChanged)
        connectMulti(self.editDuration.valueChanged, [float, Multivalue],
                     self.durationChanged)
        connectMulti(self.editDuration.valueChangedByUser, [float, Multivalue],
                     self.durationChangedByUser)
        connectMulti(self.editMaxIntensity.valueChanged, [float, Multivalue],
                     self.maxIntensityChanged)
        self.editIlluminationPattern.valueChanged.connect(self.illuminationPatternChanged)

        self.btnMoveLeft.clicked.connect(self.moveLeftClicked)
        self.btnMoveRight.clicked.connect(self.moveRightClicked)

        # Initialize presenter
        self._presenter = PulsePropertiesPresenter(self)

    def setEditWavelengthEnabled(self, enabled: bool) -> None:
        """ Sets whether the field for editing the wavelength is enabled. """
        self.editWavelength.setEnabled(enabled)
        self.editWavelength.allowSetList = enabled
    
    def setChangeOrderVisible(self, visible: bool) -> None:
        """
        Sets whether the buttons for changing the pulse's position in the scheme are visible.
        """
        setFormLayoutRowVisibility(
            self.frmBasicProperties, 3, self.lblOrder, self.editOrderContainer, visible=visible
        )

    def value(self) -> Pulse:
        return self._presenter.model

    def setValue(self, model: Pulse) -> None:
        self._presenter.model = model

    def updateIlluminationPattern(self, pattern: Pattern) -> None:
        self.editIlluminationPattern.setValue(pattern, emitSignal=False)

    def updateBasicFields(self, model: Pulse) -> None:
        self.editWavelength.setValue(model.wavelength)
        self.editDuration.setValue(model.duration)
        self.editMaxIntensity.setValue(model.max_intensity)
