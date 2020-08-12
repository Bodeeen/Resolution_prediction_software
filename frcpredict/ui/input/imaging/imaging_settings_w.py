from PyQt5.QtCore import pyqtSignal, pyqtSlot

from frcpredict.model import (
    ImagingSystemSettings, RefractiveIndex, Pattern, PatternType, Multivalue
)
from frcpredict.ui import BaseWidget
from frcpredict.ui.util import setTabOrderForChildren, connectMulti, getEnumEntryName, UserFileDirs
from .imaging_settings_p import ImagingSystemSettingsPresenter


class ImagingSystemSettingsWidget(BaseWidget):
    """
    A widget where the user may set imaging system settings.
    """

    # Signals
    valueChanged = pyqtSignal(ImagingSystemSettings)
    opticalPsfChanged = pyqtSignal(Pattern)
    pinholeFunctionChanged = pyqtSignal(Pattern)
    scanningStepSizeChanged = pyqtSignal([float], [Multivalue])
    refractiveIndexChanged = pyqtSignal(float)

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

        # Prepare UI elements
        self.presetPicker.setModelType(ImagingSystemSettings)
        self.presetPicker.setStartDirectory(UserFileDirs.ImagingSystemSettings)
        self.presetPicker.setValueGetter(self.value)
        self.presetPicker.setValueSetter(self.setValue)

        self.editOpticalPsf.setFieldName("Optical PSF")
        self.editOpticalPsf.setAllowEditGenerationAmplitude(True)
        self.editOpticalPsf.setAvailableGenerationTypes(
            [PatternType.airy_from_NA]
        )

        self.editPinholeFunction.setFieldName("Pinhole Function")
        self.editPinholeFunction.setAllowEditGenerationAmplitude(True)
        self.editPinholeFunction.setAvailableGenerationTypes(
            [PatternType.digital_pinhole, PatternType.physical_pinhole]
        )

        for refractiveIndex in list(RefractiveIndex):
            self.editImmersionType.addItem(getEnumEntryName(refractiveIndex), refractiveIndex.value)

        setTabOrderForChildren(self, [self.presetPicker, self.editOpticalPsf,
                                      self.editPinholeFunction, self.editScanningStepSize,
                                      self.editImmersionType])

        # Connect forwarded signals
        self.editOpticalPsf.valueChanged.connect(self.opticalPsfChanged)
        self.editPinholeFunction.valueChanged.connect(self.pinholeFunctionChanged)
        connectMulti(self.editScanningStepSize.valueChanged, [float, Multivalue],
                     self.scanningStepSizeChanged)
        self.editImmersionType.currentIndexChanged.connect(self._onImmersionTypeChange)

        # Initialize presenter
        self._presenter = ImagingSystemSettingsPresenter(self)

    def value(self) -> ImagingSystemSettings:
        return self._presenter.model

    def setValue(self, model: ImagingSystemSettings, emitSignal: bool = True) -> None:
        self.presetPicker.setLoadedPath(None)
        self._presenter.model = model

        if emitSignal:
            self.valueChanged.emit(model)

    def updateOpticalPsf(self, optical_psf: Pattern) -> None:
        self.editOpticalPsf.setValue(optical_psf)

    def updatePinholeFunction(self, pinhole_function: Pattern) -> None:
        self.editPinholeFunction.setValue(pinhole_function)

    def updateBasicFields(self, model: ImagingSystemSettings) -> None:
        self.editScanningStepSize.setValue(model.scanning_step_size)

        dataIndex = self.editImmersionType.findData(model.refractive_index)
        self.editImmersionType.setCurrentIndex(dataIndex)

    # Event handling
    @pyqtSlot(int)
    def _onImmersionTypeChange(self, index: int) -> None:
        self.refractiveIndexChanged.emit(self.editImmersionType.itemData(index))
