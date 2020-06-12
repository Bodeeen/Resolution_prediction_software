from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap

from frcpredict.model import ImagingSystemSettings
from frcpredict.ui import BaseWidget
from .imaging_settings_p import ImagingSystemSettingsPresenter


class ImagingSystemSettingsWidget(BaseWidget):
    """
    A widget where the user may set imaging system settings.
    """

    # Signals
    loadOpticalPsfClicked = pyqtSignal()
    loadPinholeFunctionClicked = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)
        
        # Connect forwarded signals
        self.btnLoadOpticalPsf.clicked.connect(self.loadOpticalPsfClicked)
        self.btnLoadPinholeFunction.clicked.connect(self.loadPinholeFunctionClicked)

        # Initialize presenter
        self._presenter = ImagingSystemSettingsPresenter(self)

    def setModel(self, model: ImagingSystemSettings) -> None:
        self._presenter.model = model

    def setOpticalPsfPixmap(self, pixmap: QPixmap) -> None:
        self.imgOpticalPsf.setPixmap(pixmap)

    def setPinholeFunctionPixmap(self, pixmap: QPixmap) -> None:
        self.imgPinholeFunction.setPixmap(pixmap)

    def updateBasicFields(self, model: ImagingSystemSettings) -> None:
        self.editScanningStepSize.setValue(model.scanning_step_size)
