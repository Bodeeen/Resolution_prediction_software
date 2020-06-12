from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap

from frcpredict.model import Pulse
from frcpredict.ui import BaseWidget
from frcpredict.util import patterns
from .pulse_properties_p import PulsePropertiesPresenter


class PulsePropertiesWidget(BaseWidget):
    """
    A widget where the user may set propreties of a specific pulse.
    """

    # Signals
    wavelengthChanged = pyqtSignal(int)
    durationChanged = pyqtSignal(float)
    maxIntensityChanged = pyqtSignal(float)
    patternSelectionChanged = pyqtSignal(int)
    moveLeftClicked = pyqtSignal()
    moveRightClicked = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

        for pattern in patterns.keys():
            self.listPatterns.addItem(pattern)
        
        # Connect forwarded signals
        self.editWavelength.valueChanged.connect(self.wavelengthChanged)
        self.editDuration.valueChanged.connect(self.durationChanged)
        self.editMaxIntensity.valueChanged.connect(self.maxIntensityChanged)
        self.listPatterns.currentRowChanged.connect(self.patternSelectionChanged)
        self.btnMoveLeft.clicked.connect(self.moveLeftClicked)
        self.btnMoveRight.clicked.connect(self.moveRightClicked)

        # Initialize presenter
        self._presenter = PulsePropertiesPresenter(self)

    def setModel(self, model: Pulse) -> None:
        self._presenter.model = model

    def setEditWavelengthEnabled(self, value: bool) -> None:
        self.editWavelength.setEnabled(value)
    
    def setChangeOrderVisible(self, value: bool) -> None:
        self.lblOrder.setVisible(value)
        self.btnMoveLeft.setVisible(value)
        self.btnMoveRight.setVisible(value)

    def setIlluminationPatternPixmap(self, pixmap: QPixmap) -> None:
        self.imgIlluminationPattern.setPixmap(pixmap)

    def updateBasicFields(self, model: Pulse) -> None:
        self.editWavelength.setValue(model.wavelength)
        self.editDuration.setValue(model.duration)
        self.editMaxIntensity.setValue(model.max_intensity)
