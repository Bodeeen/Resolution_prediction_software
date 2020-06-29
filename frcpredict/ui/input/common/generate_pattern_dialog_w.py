from typing import Optional, Tuple, List

from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QDialog, QDialogButtonBox

from frcpredict.model import Pattern, PatternType
from frcpredict.ui import BaseWidget
from frcpredict.ui.util import getPatternTypeName, setFormLayoutRowVisibility
from frcpredict.util import with_cleared_signals
from .generate_pattern_dialog_p import GeneratePatternPresenter
from .list_item_with_value import ListItemWithValue


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

    # Methods
    def __init__(self, parent: Optional[QWidget] = None, title: str = "Generate Pattern",
                 availableTypes: List[PatternType] = [], allowEditAmplitude: bool = True,
                 normalisePreview: bool = False) -> None:
        self._hasHandledInitialRowChange = False
        self._allowEditAmplitude = allowEditAmplitude

        super().__init__(__file__, parent, Qt.WindowSystemMenuHint | Qt.WindowTitleHint)
        self.setWindowTitle(title)
        self.setAvailableTypes(availableTypes)
        self._updateOKButton()

        # Connect own signals
        self.listType.currentRowChanged.connect(self._onTypeListRowChange)

        # Connect forwarded signals
        self.editAmplitude.valueChanged.connect(self.amplitudeChanged)
        self.editRadius.valueChanged.connect(self.radiusChanged)
        self.editFwhm.valueChanged.connect(self.fwhmChanged)
        self.editPeriodicity.valueChanged.connect(self.periodicityChanged)

        # Initialize presenter
        self._presenter = GeneratePatternPresenter(self, normalisePreview)

    def setAvailableTypes(self, patternTypes: List[PatternType]):
        """ Sets which pattern types are available for the user to pick. """
        self.listType.clear()
        for patternType in patternTypes:
            self.listType.addItem(
                ListItemWithValue(text=getPatternTypeName(patternType), value=patternType)
            )

    def setAvailableProperties(self, amplitude: bool = False, radius: bool = False,
                               fwhm: bool = False, periodicity: bool = False) -> None:
        """ Sets which pattern properties are available for the user to modify. """

        setFormLayoutRowVisibility(self.frmProperties, 0, self.lblAmplitude, self.editAmplitude,
                                   visible=amplitude and self._allowEditAmplitude)

        setFormLayoutRowVisibility(self.frmProperties, 1, self.lblRadius, self.editRadius,
                                   visible=radius)

        setFormLayoutRowVisibility(self.frmProperties, 2, self.lblFwhm, self.editFwhm,
                                   visible=fwhm)

        setFormLayoutRowVisibility(self.frmProperties, 3, self.lblPeriodicity, self.editPeriodicity,
                                   visible=periodicity)

    def value(self) -> Pattern:
        return self._presenter.model

    def setValue(self, model: Pattern) -> None:
        self._presenter.model = model

    def updatePreview(self, pixmap: QPixmap) -> None:
        self.imgPreview.setPixmap(pixmap)

    def updatePropertyFields(self, amplitude: Optional[float] = None,
                             radius: Optional[float] = None, fwhm: Optional[float] = None,
                             periodicity: Optional[float] = None) -> None:
        """ Updates the values of the fields in the widget. """

        if amplitude is not None:
            self.editAmplitude.setValue(amplitude)

        if radius is not None:
            self.editRadius.setValue(radius)

        if fwhm is not None:
            self.editFwhm.setValue(fwhm)

        if periodicity is not None:
            self.editPeriodicity.setValue(periodicity)

    @staticmethod
    def getPatternData(parent: Optional[QWidget] = None, title: str = "Generate Pattern",
                       availableTypes: List[PatternType] = [], allowEditAmplitude: bool = True,
                       normalisePreview: bool = False) -> Tuple[Optional[Pattern], bool]:
        """
        Synchronously opens a dialog for entering pattern properties. The second value in the
        returned tuple refers to whether the "OK" button was pressed when the dialog closed. If
        it's true, the first value will contain the pattern data.
        """

        dialog = GeneratePatternDialog(parent, title, availableTypes,
                                       allowEditAmplitude, normalisePreview)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            pattern_data = with_cleared_signals(dialog.value().pattern_data)
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
    def _onTypeListRowChange(self, selectedRow: int):
        if not self._hasHandledInitialRowChange and selectedRow > -1:
            # We do this to make sure no row is selected when the dialog opens
            self.listType.setCurrentRow(-1)
            self._hasHandledInitialRowChange = True
            self.typeChanged.emit(None)
            return

        if selectedRow > -1:
            self.typeChanged.emit(self.listType.item(selectedRow).value())
        else:
            self.typeChanged.emit(None)

        self._updateOKButton()
