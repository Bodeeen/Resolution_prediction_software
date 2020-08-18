from copy import deepcopy
from typing import Optional, Tuple, List

from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QDialog, QDialogButtonBox

from frcpredict.model import SampleStructure, SampleStructureProperties
from frcpredict.ui import BaseWidget, ListItemWithValue
from frcpredict.ui.util import snakeCaseToName
from frcpredict.util import clear_signals
from .sample_structure_picker_dialog_p import SampleStructurePickerPresenter


class SampleStructurePickerDialog(QDialog, BaseWidget):
    """
    A dialog for selecting a predefined sample structure.
    """

    # Signals
    valueChanged = pyqtSignal(object)

    # Methods
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self._hasHandledInitialRowChange = False

        super().__init__(__file__, parent, Qt.WindowSystemMenuHint | Qt.WindowTitleHint)
        self._updateOKButton()

        # Connect own signal slots
        self.listSample.currentRowChanged.connect(self._onSampleListRowChange)

        # Initialize presenter
        self._presenter = SampleStructurePickerPresenter(self)

    def setAvailableStructures(self, structures: List[SampleStructure]):
        """ Sets which sample structures are available for the user to pick. """
        self.listSample.clear()
        for structure in structures:
            listItemText = structure.properties.name or snakeCaseToName(structure.image.id)
            self.listSample.addItem(
                ListItemWithValue(text=listItemText, value=structure, tag=structure.image.id)
            )

    def value(self) -> SampleStructure:
        return self._presenter.model

    def setValue(self, model: SampleStructure) -> None:
        self._presenter.model = model
        self._hasHandledInitialRowChange = True

    def updatePreview(self, pixmap: QPixmap) -> None:
        """ Updates the preview in the widget. """
        self.imgPreview.setPixmap(pixmap)

    def updateStructure(self, structureId: str, properties: SampleStructureProperties) -> None:
        """ Updates the selected structure in the widget. """

        self.editSpectralPower.setValue(properties.spectral_power)
        self.editKOrigin.setValue(properties.K_origin)

        self.listSample.blockSignals(True)
        try:
            for i in range(0, self.listSample.count()):
                if self.listSample.item(i).tag() == structureId:
                    self.listSample.setCurrentRow(i)
                    return

            self.listSample.setCurrentRow(-1)  # Unselect if no match found in sample list
        finally:
            self.listSample.blockSignals(False)
            self._updateOKButton()

    @staticmethod
    def getSampleStructure(parent: Optional[QWidget] = None,
                           initialValue: Optional[SampleStructure] = None) -> Tuple[Optional[SampleStructure], bool]:
        """
        Synchronously opens a dialog for picking a sample structure. The second value in the
        returned tuple refers to whether the "OK" button was pressed when the dialog closed. If
        it's true, the first value will contain the picked sample structure.
        """

        dialog = SampleStructurePickerDialog(parent)

        if initialValue is not None:
            dialog.setValue(clear_signals(deepcopy(initialValue)))

        result = dialog.exec_()

        if result == QDialog.Accepted:
            sampleStructure = clear_signals(dialog.value())
        else:
            sampleStructure = None

        dialog.deleteLater()  # Prevent memory leak
        return sampleStructure, result == QDialog.Accepted

    # Internal methods
    def _updateOKButton(self) -> None:
        """ Enables or disables the "OK" button depending on whether the entered data is valid. """
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(
            self.listSample.currentRow() > -1
        )

    # Event handling
    @pyqtSlot(int)
    def _onSampleListRowChange(self, selectedRow: int) -> None:
        if not self._hasHandledInitialRowChange and selectedRow > -1:
            # We do this to make sure no row is selected when the dialog opens
            self.listSample.setCurrentRow(-1)
            self._hasHandledInitialRowChange = True
            return

        if selectedRow > -1:
            self.valueChanged.emit(self.listSample.item(selectedRow).value())
        else:
            self.valueChanged.emit(None)

        self._updateOKButton()
