from copy import deepcopy
from typing import Optional, Tuple

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QDialog, QDialogButtonBox

from frcpredict.model import DisplayableSample, SampleStructure
from frcpredict.ui import BaseWidget
from frcpredict.util import clear_signals
from .sample_structure_picker_dialog_m import SampleStructurePickerModel, SampleType
from .sample_structure_picker_dialog_p import SampleStructurePickerPresenter


class SampleStructurePickerDialog(QDialog, BaseWidget):
    """
    A dialog for selecting a predefined sample structure.
    """

    # Signals
    generateRandomClicked = pyqtSignal()
    generatePairsClicked = pyqtSignal()
    generateLinesClicked = pyqtSignal()
    loadFromInputClicked = pyqtSignal()
    loadFromFileClicked = pyqtSignal()

    # Methods
    def __init__(self, parent: Optional[QWidget] = None,
                 inputSampleStructure: Optional[SampleStructure] = None,
                 forOutput: bool = False, allowLoadFile: bool = False) -> None:
        super().__init__(__file__, parent, Qt.WindowSystemMenuHint | Qt.WindowTitleHint)

        # Prepare UI elements
        self.btnLoadFromInput.setVisible(forOutput)
        self.btnLoadFromInput.setEnabled(inputSampleStructure is not None)
        self.btnLoadFromFile.setVisible(allowLoadFile)

        if forOutput:
            if inputSampleStructure is not None:
                self.lblInfo.setText("Note: FRC/resolution output data may not be accurate when" +
                                     " picking anything other than the sample structure loaded" +
                                     " in the input.")
            else:
                self.lblInfo.setText("Note: FRC/resolution output data may not be accurate, since" +
                                     " the simulation was run without any loaded sample structure.")

        # Connect forwarded signals
        self.btnGenerateRandom.clicked.connect(self.generateRandomClicked)
        self.btnGeneratePairs.clicked.connect(self.generatePairsClicked)
        self.btnGenerateLines.clicked.connect(self.generateLinesClicked)
        self.btnLoadFromInput.clicked.connect(self.loadFromInputClicked)
        self.btnLoadFromFile.clicked.connect(self.loadFromFileClicked)

        # Initialize presenter
        self._presenter = SampleStructurePickerPresenter(self, inputSampleStructure)

        self._updateOKButton()

    def value(self) -> SampleStructurePickerModel:
        return self._presenter.model

    def setValue(self, model: SampleStructurePickerModel) -> None:
        self._presenter.model = model

    def updateLoadedType(self, sampleType: SampleType) -> None:
        selectedStyleSheet = """
            QPushButton {
                border: 2px solid green;
                background-color: #E0E0E0;
            }
            
            QPushButton:hover {
                border: 2px solid #00B000;
                background-color: #E8E8E8;
            }
            
            QPushButton:pressed {
                border: 2px solid black;
            }
            """

        self.btnLoadFromInput.setStyleSheet("")
        self.btnGenerateRandom.setStyleSheet("")
        self.btnGeneratePairs.setStyleSheet("")
        self.btnGenerateLines.setStyleSheet("")
        self.btnLoadFromFile.setStyleSheet("")

        if sampleType == SampleType.fromInput:
            self.btnLoadFromInput.setStyleSheet(selectedStyleSheet)
        elif sampleType == SampleType.random:
            self.btnGenerateRandom.setStyleSheet(selectedStyleSheet)
        elif sampleType == SampleType.pairs:
            self.btnGeneratePairs.setStyleSheet(selectedStyleSheet)
        elif sampleType == SampleType.lines:
            self.btnGenerateLines.setStyleSheet(selectedStyleSheet)
        elif sampleType == SampleType.file:
            self.btnLoadFromFile.setStyleSheet(selectedStyleSheet)

        try:
            self._updateOKButton()
        except AttributeError:
            pass

    def updatePreview(self, pixmap: QPixmap) -> None:
        """ Updates the preview in the widget. """
        self.imgPreview.setPixmap(pixmap)

    @staticmethod
    def getSampleStructure(parent: Optional[QWidget] = None) -> Tuple[Optional[SampleStructure], bool]:
        """
        Synchronously opens a dialog for picking a sample structure. The second value in the
        returned tuple refers to whether the "OK" button was pressed when the dialog closed. If
        it's true, the first value will contain the picked sample structure.
        """
        dialog = SampleStructurePickerDialog(parent)
        sampleStructure, result = SampleStructurePickerDialog._execDialog(dialog)
        assert sampleStructure is None or isinstance(sampleStructure, SampleStructure)
        return sampleStructure, result

    @staticmethod
    def getDisplayableSampleForOutput(parent: Optional[QWidget] = None,
                                      inputSampleStructure: Optional[SampleStructure] = None) -> Tuple[Optional[DisplayableSample], bool]:
        """
        Synchronously opens a dialog for picking a displayable sample. The second value in the
        returned tuple refers to whether the "OK" button was pressed when the dialog closed. If
        it's true, the first value will contain the picked displayable sample.
        """
        dialog = SampleStructurePickerDialog(parent, inputSampleStructure,
                                             forOutput=True, allowLoadFile=True)
        return SampleStructurePickerDialog._execDialog(dialog)

    # Internal methods
    def _updateOKButton(self) -> None:
        """ Enables or disables the "OK" button depending on whether the entered data is valid. """
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(
            self.value().isSampleLoaded()
        )

    @staticmethod
    def _execDialog(dialog: "SampleStructurePickerDialog") -> Tuple[Optional[DisplayableSample], bool]:
        result = dialog.exec_()

        if result == QDialog.Accepted:
            displayableSample, _ = dialog.value().getLoadedSample()
            clear_signals(displayableSample)
        else:
            displayableSample = None

        dialog.deleteLater()  # Prevent memory leak
        return displayableSample, result == QDialog.Accepted
