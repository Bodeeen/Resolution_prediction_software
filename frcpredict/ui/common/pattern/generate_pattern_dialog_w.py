from copy import deepcopy
from typing import Optional, Tuple, List

from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QDialog, QDialogButtonBox

from frcpredict.model import Pattern, PatternType
from frcpredict.ui import BaseWidget
from frcpredict.ui.util import getEnumEntryName, setFormLayoutRowVisibility, setTabOrderForChildren
from frcpredict.util import clear_signals
from ..list_item_with_value import ListItemWithValue
from .generate_pattern_dialog_p import GeneratePatternPresenter


class GeneratePatternDialog(QDialog, BaseWidget):
    """
    A dialog for generating patterns.
    """

    # Signals
    typeChanged = pyqtSignal(object)
    amplitudeChanged = pyqtSignal(float)
    radiusChanged = pyqtSignal(float)
    fwhmChanged = pyqtSignal(float)
    periodicityChanged = pyqtSignal(float)
    zeroIntensityChanged = pyqtSignal(float)
    naChanged = pyqtSignal(float)
    emissionWavelengthChanged = pyqtSignal(float)

    # Methods
    def __init__(self, parent: Optional[QWidget] = None, title: str = "Generate Pattern",
                 availableTypes: List[PatternType] = [], allowEditAmplitude: bool = True,
                 canvasInnerRadiusNm: float = 500.0, normalizePreview: bool = False) -> None:
        self._allowEditAmplitude = allowEditAmplitude
        self._canvasInnerRadius = canvasInnerRadiusNm
        self._hasHandledInitialRowChange = False

        super().__init__(__file__, parent, Qt.WindowSystemMenuHint | Qt.WindowTitleHint)

        # Prepare UI elements
        self.setWindowTitle(title)
        self.setAvailableTypes(availableTypes)
        self._updateOKButton()

        setTabOrderForChildren(self, [self.listType, self.editAmplitude, self.editRadius,
                                      self.editFwhm, self.editPeriodicity, self.editZeroIntensity,
                                      self.editNa, self.editEmissionWavelength])

        # Connect own signal slots
        self.listType.currentRowChanged.connect(self._onTypeListRowChange)

        # Connect forwarded signals
        self.editAmplitude.valueChanged.connect(self.amplitudeChanged)
        self.editRadius.valueChanged.connect(self.radiusChanged)
        self.editFwhm.valueChanged.connect(self.fwhmChanged)
        self.editPeriodicity.valueChanged.connect(self.periodicityChanged)
        self.editZeroIntensity.valueChanged.connect(self.zeroIntensityChanged)
        self.editNa.valueChanged.connect(self.naChanged)
        self.editEmissionWavelength.valueChanged.connect(self.emissionWavelengthChanged)

        # Initialize presenter
        self._presenter = GeneratePatternPresenter(self, normalizePreview)

    def setAvailableTypes(self, patternTypes: List[PatternType]):
        """ Sets which pattern types are available for the user to pick. """
        self.listType.clear()
        for patternType in patternTypes:
            self.listType.addItem(
                ListItemWithValue(text=getEnumEntryName(patternType), value=patternType)
            )

    def setAvailableProperties(self, amplitude: bool = False, radius: bool = False,
                               fwhm: bool = False, periodicity: bool = False,
                               zeroIntensity: bool = False, na: bool = False,
                               emissionWavelength: bool = False) -> None:
        """ Sets which pattern properties are available for the user to modify. """

        setFormLayoutRowVisibility(self.frmProperties, 0, self.lblAmplitude, self.editAmplitude,
                                   visible=amplitude and self._allowEditAmplitude)

        setFormLayoutRowVisibility(self.frmProperties, 1, self.lblRadius, self.editRadius,
                                   visible=radius)

        setFormLayoutRowVisibility(self.frmProperties, 2, self.lblFwhm, self.editFwhm,
                                   visible=fwhm)

        setFormLayoutRowVisibility(self.frmProperties, 3, self.lblPeriodicity, self.editPeriodicity,
                                   visible=periodicity)

        setFormLayoutRowVisibility(self.frmProperties, 4,
                                   self.lblZeroIntensity, self.editZeroIntensity,
                                   visible=zeroIntensity)

        setFormLayoutRowVisibility(self.frmProperties, 5, self.lblNa, self.editNa,
                                   visible=na)

        setFormLayoutRowVisibility(self.frmProperties, 6,
                                   self.lblEmissionWavelength, self.editEmissionWavelength,
                                   visible=emissionWavelength)

    def canvasInnerRadius(self) -> float:
        """
        Sets the inner radius of the canvas, in nanometres, that will be used to generate pattern
        previews.
        """
        return self._canvasInnerRadius

    def value(self) -> Pattern:
        return self._presenter.model

    def setValue(self, model: Pattern) -> None:
        self._presenter.model = model
        self._hasHandledInitialRowChange = True

    def updatePreview(self, pixmap: QPixmap) -> None:
        """ Updates the preview in the widget. """
        self.imgPreview.setPixmap(pixmap)

    def updateType(self, patternType: PatternType) -> None:
        """ Updates the selected pattern type in the widget. """

        self.listType.blockSignals(True)
        try:
            for i in range(0, self.listType.count()):
                if self.listType.item(i).value() == patternType:
                    self.listType.setCurrentRow(i)
                    return

            self.listType.setCurrentRow(-1)  # Unselect if no match found in type list
        finally:
            self.listType.blockSignals(False)
            self._updateOKButton()

    def updatePropertyFields(self, amplitude: Optional[float] = None,
                             radius: Optional[float] = None, fwhm: Optional[float] = None,
                             periodicity: Optional[float] = None,
                             zeroIntensity: Optional[float] = None, na: Optional[float] = None,
                             emissionWavelength: Optional[float] = None) -> None:
        """ Updates the values of the fields in the widget. """

        if amplitude is not None:
            self.editAmplitude.setValue(amplitude)

        if radius is not None:
            self.editRadius.setValue(radius)

        if fwhm is not None:
            self.editFwhm.setValue(fwhm)

        if periodicity is not None:
            self.editPeriodicity.setValue(periodicity)

        if zeroIntensity is not None:
            self.editZeroIntensity.setValue(zeroIntensity)

        if na is not None:
            self.editNa.setValue(na)

        if emissionWavelength is not None:
            self.editEmissionWavelength.setValue(emissionWavelength)

    @staticmethod
    def getPatternData(parent: Optional[QWidget] = None, title: str = "Generate Pattern",
                       availableTypes: List[PatternType] = [], allowEditAmplitude: bool = True,
                       canvasInnerRadiusNm: float = 500.0, normalizePreview: bool = False,
                       initialValue: Optional[Pattern] = None) -> Tuple[Optional[Pattern], bool]:
        """
        Synchronously opens a dialog for entering pattern properties. The second value in the
        returned tuple refers to whether the "OK" button was pressed when the dialog closed. If
        it's true, the first value will contain the pattern data.
        """

        dialog = GeneratePatternDialog(parent, title, availableTypes,
                                       allowEditAmplitude, canvasInnerRadiusNm, normalizePreview)

        if initialValue is not None and initialValue.pattern_type != PatternType.array2D:
            dialog.setValue(clear_signals(deepcopy(initialValue)))

        result = dialog.exec_()

        if result == QDialog.Accepted:
            pattern_data = clear_signals(dialog.value().pattern_data)
        else:
            pattern_data = None

        dialog.deleteLater()  # Prevent memory leak
        return pattern_data, result == QDialog.Accepted

    # Internal methods
    def _updateOKButton(self) -> None:
        """ Enables or disables the "OK" button depending on whether the entered data is valid. """
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(
            self.listType.currentRow() > -1
        )

    # Event handling
    @pyqtSlot(int)
    def _onTypeListRowChange(self, selectedRow: int) -> None:
        if not self._hasHandledInitialRowChange and selectedRow > -1:
            # We do this to make sure no row is selected when the dialog opens
            self.listType.setCurrentRow(-1)
            self._hasHandledInitialRowChange = True
            return

        if selectedRow > -1:
            self.typeChanged.emit(self.listType.item(selectedRow).value())
        else:
            self.typeChanged.emit(None)

        self._updateOKButton()
