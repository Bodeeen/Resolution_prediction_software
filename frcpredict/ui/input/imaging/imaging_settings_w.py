from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap

from frcpredict.model import ImagingSystemSettings, Pattern, PatternType
from frcpredict.ui import BaseWidget
from .imaging_settings_p import ImagingSystemSettingsPresenter


class ImagingSystemSettingsWidget(BaseWidget):
    """
    A widget where the user may set imaging system settings.
    """

    # Signals
    scanningStepSizeChanged = pyqtSignal(float)

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

        self.editOpticalPsf.setFieldName("Optical PSF")
        self.editOpticalPsf.setAllowEditGenerationAmplitude(True)
        self.editOpticalPsf.setAvailableGenerationTypes(
            [PatternType.gaussian, PatternType.airy]
        )

        self.editPinholeFunction.setFieldName("Pinhole Function")
        self.editPinholeFunction.setAllowEditGenerationAmplitude(True)
        self.editPinholeFunction.setAvailableGenerationTypes(
            [PatternType.digital_pinhole]
        )

        # Connect forwarded signals
        self.editScanningStepSize.valueChanged.connect(self.scanningStepSizeChanged)

        # Initialize presenter
        self._presenter = ImagingSystemSettingsPresenter(self)

    def setValue(self, model: ImagingSystemSettings) -> None:
        self._presenter.model = model

    def updateOpticalPsf(self, optical_psf: Pattern) -> None:
        self.editOpticalPsf.setValue(optical_psf)

    def updatePinholeFunction(self, pinhole_function: Pattern) -> None:
        self.editPinholeFunction.setValue(pinhole_function)

    def updateBasicFields(self, model: ImagingSystemSettings) -> None:
        self.editScanningStepSize.setValue(model.scanning_step_size)
