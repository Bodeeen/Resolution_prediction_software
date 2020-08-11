from typing import Optional

import numpy as np
from PyQt5.QtCore import pyqtSignal

from frcpredict.ui import BaseWidget
from ..controls.output_director_w import OutputDirectorWidget
from .virtual_imaging_results_m import VirtualImagingResultsModel
from .virtual_imaging_results_p import VirtualImagingResultsPresenter


class VirtualImagingResultsWidget(BaseWidget):
    """
    A widget that displays an expected image generated by simulation. Contains controls for loading
    a sample image to generate the expected image from as well as for setting pan/zoom and
    auto-value settings.
    """

    # Signals
    expectedImageChanged = pyqtSignal(object, object, bool)

    loadImageClicked = pyqtSignal()
    unloadImageClicked = pyqtSignal()
    exportImageClicked = pyqtSignal()

    panZoomResetClicked = pyqtSignal()
    panZoomAutoResetChanged = pyqtSignal(int)

    autoLevelClicked = pyqtSignal()
    autoLevelAutoPerformChanged = pyqtSignal(int)
    autoLevelLowerCutoffChanged = pyqtSignal(float)

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        self._outputDirector = None
        self._imageIsSet = False
        super().__init__(__file__, *args, **kwargs)

        # Prepare UI elements
        self.imgExpectedImage.ui.roiBtn.setVisible(False)
        self.imgExpectedImage.ui.menuBtn.setVisible(False)

        # Connect forwarded signals
        self.btnLoadImage.clicked.connect(self.loadImageClicked)
        self.btnUnloadImage.clicked.connect(self.unloadImageClicked)
        self.btnExportImage.clicked.connect(self.exportImageClicked)

        self.btnResetPanZoom.clicked.connect(self.panZoomResetClicked)
        self.chkPanZoomAutoReset.stateChanged.connect(self.panZoomAutoResetChanged)

        self.btnAutoLevel.clicked.connect(self.autoLevelClicked)
        self.chkAutoLevelAutoPerform.stateChanged.connect(self.autoLevelAutoPerformChanged)
        self.editAutoLevelLowerCutoff.valueChanged.connect(self.autoLevelLowerCutoffChanged)

        # Initialize presenter
        self._presenter = VirtualImagingResultsPresenter(self)

    def resetPanZoom(self) -> None:
        """ Resets the expected image to be centered and fitted within the image view. """
        self.imgExpectedImage.autoRange()

    def autoLevel(self, lowerCutoff: float = 0.0) -> None:
        """
        Automatically adjusts the min/max levels in the image view to match the image min/max
        values.
        """
        self.imgExpectedImage.autoLevels()
        self.cutoffLower(lowerCutoff)

    def cutoffLower(self, cutoff: float = 0.0) -> None:
        """ Sets the min level in the image view to the given value. """
        self.imgExpectedImage.setLevels(min=cutoff, max=self.imgExpectedImage.getLevels()[1])

    def outputDirector(self) -> Optional[OutputDirectorWidget]:
        """ Returns the output director that this widget will receive its data from. """
        return self._outputDirector

    def setOutputDirector(self, outputDirector: Optional[OutputDirectorWidget]) -> None:
        """ Sets the output director that this widget will receive its data from. """

        if self._outputDirector is not None:
            self._outputDirector.expectedImageChanged.disconnect(self.expectedImageChanged)

        if outputDirector is not None:
            outputDirector.expectedImageChanged.connect(self.expectedImageChanged)

        self._outputDirector = outputDirector

    def updateBasicFields(self, model: VirtualImagingResultsModel) -> None:
        self.chkPanZoomAutoReset.setChecked(model.panZoomAutoReset)
        self.chkAutoLevelAutoPerform.setChecked(model.autoLevelAutoPerform)
        self.editAutoLevelLowerCutoff.setValue(model.autoLevelLowerCutoff)

    def updateExpectedImage(self, expectedImage: np.ndarray, autoRange: bool, autoLevel: bool,
                            autoLevelLowerCutoff: float = 0.0) -> None:
        """ Updates the expected image in the widget. """

        self.btnLoadImage.setEnabled(True)

        imageIsLoaded = expectedImage is not None and len(expectedImage.shape) > 0

        if imageIsLoaded:
            self.imgExpectedImage.setImage(expectedImage.T,
                                           autoRange=autoRange,
                                           autoLevels=autoLevel)

            if autoLevel:
                self.imgExpectedImage.setLevels(min=autoLevelLowerCutoff,
                                                max=self.imgExpectedImage.getLevels()[1])
        elif self._imageIsSet:
            self.imgExpectedImage.setImage(np.zeros((0, 0)))

        self.imgExpectedImage.setEnabled(imageIsLoaded)
        self.btnExportImage.setEnabled(imageIsLoaded)
        self.grpPanZoom.setEnabled(imageIsLoaded)
        self.grpAutoLevelling.setEnabled(imageIsLoaded)
        self.btnUnloadImage.setEnabled(imageIsLoaded)

        self._imageIsSet = imageIsLoaded
