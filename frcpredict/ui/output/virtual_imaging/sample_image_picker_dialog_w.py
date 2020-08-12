from typing import Optional, Tuple, List

from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QDialog, QDialogButtonBox

from frcpredict.model import SampleStructure
from frcpredict.ui import BaseWidget, ListItemWithValue
from frcpredict.ui.util import snakeCaseToName
from frcpredict.util import clear_signals
from .sample_image_picker_dialog_m import SampleImagePickerModel
from .sample_image_picker_dialog_p import SampleImagePickerPresenter


class SampleImagePickerDialog(QDialog, BaseWidget):
    """
    A dialog for picking a sample image.
    """

    # Signals
    sampleStructurePicked = pyqtSignal(object)
    fromSampleSelected = pyqtSignal()
    fromFileSelected = pyqtSignal()
    loadFileClicked = pyqtSignal()

    # Methods
    def __init__(self, parent: Optional[QWidget] = None,
                 loadedSampleStructureId: Optional[str] = None) -> None:
        self._loadedSampleStructureId = loadedSampleStructureId

        super().__init__(__file__, parent, Qt.WindowSystemMenuHint | Qt.WindowTitleHint)
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(False)

        # Connect own signal slots
        self.listSample.currentRowChanged.connect(self._onSampleListRowChange)

        # Connect forwarded signals
        self.rdoSample.clicked.connect(self.fromSampleSelected)
        self.rdoFile.clicked.connect(self.fromFileSelected)
        self.btnLoadFile.clicked.connect(self.loadFileClicked)

        # Initialize presenter
        self._presenter = SampleImagePickerPresenter(self)

    def setAvailableSampleStructures(self, samples: List[SampleStructure]):
        """
        Sets which sample structures are available for the user to pick. If a sample structure was
        loaded for simulation, it will be pre-selected here.
        """

        loadedSampleIndex = -1

        self.listSample.clear()
        for sampleStructureIndex, sampleStructure in enumerate(samples):
            listItemText = snakeCaseToName(sampleStructure.id)
            if sampleStructure.id == self._loadedSampleStructureId:
                listItemText += " (loaded)"  # Indicate that this is the loaded sample structure
                loadedSampleIndex = sampleStructureIndex

            self.listSample.addItem(
                ListItemWithValue(text=listItemText, value=sampleStructure, tag=sampleStructure.id)
            )

        if loadedSampleIndex > -1:
            self.listSample.setCurrentRow(loadedSampleIndex)
            self.lblInfo.setText("Note: FRC/resolution output data will not be accurate when" +
                                 " picking anything other than the loaded sample structure.")
        else:
            self.lblInfo.setText("Note: FRC/resolution output data will not be accurate, since" +
                                 " the simulation was run without any loaded sample structure.")

    def value(self) -> SampleImagePickerModel:
        return self._presenter.model

    def setValue(self, model: SampleImagePickerModel) -> None:
        self._presenter.model = model
        self._hasHandledInitialRowChange = True

    def updatePreview(self, pixmap: QPixmap) -> None:
        """ Updates the preview in the widget. """
        self.imgPreview.setPixmap(pixmap)

    def updateFields(self, imageLoaded: bool, willLoadFromFile: bool,
                     sampleStructureId: Optional[str] = None) -> None:
        """ Updates the fields in the widget. """

        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(imageLoaded)

        self.rdoSample.setChecked(not willLoadFromFile)
        self.rdoFile.setChecked(willLoadFromFile)
        self.listSample.setEnabled(not willLoadFromFile)
        self.btnLoadFile.setEnabled(willLoadFromFile)

        self.listSample.blockSignals(True)
        try:
            for i in range(0, self.listSample.count()):
                if self.listSample.item(i).tag() == sampleStructureId:
                    self.listSample.setCurrentRow(i)
                    return

            self.listSample.setCurrentRow(-1)  # Unselect if no match found in sample list
        finally:
            self.listSample.blockSignals(False)

    @staticmethod
    def getImageData(parent: Optional[QWidget] = None,
                     loadedSampleStructureId: Optional[str] = None) -> Tuple[Optional[SampleImagePickerModel], bool]:
        """
        Synchronously opens a dialog for picking a sample image. The second value in the returned
        tuple refers to whether the "OK" button was pressed when the dialog closed. If it's true,
        the first value will contain the image data.
        """

        dialog = SampleImagePickerDialog(parent, loadedSampleStructureId)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            imageData = clear_signals(dialog.value())
        else:
            imageData = None

        dialog.deleteLater()  # Prevent memory leak
        return imageData, result == QDialog.Accepted

    # Event handling
    @pyqtSlot(int)
    def _onSampleListRowChange(self, selectedRow: int) -> None:
        if selectedRow > -1:
            self.sampleStructurePicked.emit(self.listSample.item(selectedRow).value())
        else:
            self.sampleStructurePicked.emit(None)
