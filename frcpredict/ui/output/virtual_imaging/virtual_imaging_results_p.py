from typing import Optional

import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog
from skimage.io import imsave

from frcpredict.model import RunInstance, KernelSimulationResult, SampleImage
from frcpredict.ui import BasePresenter, Preferences, SampleStructurePickerDialog
from frcpredict.ui.util import UserFileDirs
from .virtual_imaging_results_m import VirtualImagingResultsModel


class VirtualImagingResultsPresenter(BasePresenter[VirtualImagingResultsModel]):
    """
    Presenter for the virtual imaging results widget.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: VirtualImagingResultsModel) -> None:
        # Set model
        self._model = model

        # Trigger model change event handlers
        self.widget.updateBasicFields(model)

    # Methods
    def __init__(self, widget) -> None:
        # These should really be in the model, but initialDisplayOfData prevents it
        self._sampleImageArr = None
        self._expectedImageArr = None

        super().__init__(VirtualImagingResultsModel(), widget)

        # Prepare UI events
        widget.kernelResultChanged.connect(self._uiKernelResultChange)

        widget.loadImageClicked.connect(self._uiClickLoadImage)
        widget.unloadImageClicked.connect(self._uiClickUnloadImage)
        widget.exportSampleImageClicked.connect(self._uiClickExportSampleImage)
        widget.exportExpectedImageClicked.connect(self._uiClickExportExpectedImage)

        widget.panZoomResetClicked.connect(self._uiClickPanZoomReset)
        widget.panZoomAutoResetToggled.connect(self._uiPanZoomAutoResetChange)

        widget.autoLevelClicked.connect(self._uiClickAutoLevel)
        widget.autoLevelAutoPerformToggled.connect(self._uiAutoLevelAutoPerformChange)
        widget.autoLevelLowerCutoffChanged.connect(self._uiAutoLevelLowerCutoffChange)

    # UI event handling
    @pyqtSlot(object, object, bool)
    def _uiKernelResultChange(self, runInstance: RunInstance,
                              kernelResult: Optional[KernelSimulationResult],
                              initialDisplayOfData: bool) -> None:
        """
        Retrieves the expected image for the given kernel simulation result and tells the widget to
        display it.
        """

        if self.widget.outputDirector() is None:
            return

        displayableSample = self.widget.outputDirector().value().displayableSample
        if kernelResult is not None and displayableSample is not None:
            sampleImageArr = displayableSample.get_image_arr(
                runInstance.imaging_system_settings.scanning_step_size
            )
            expectedImageArr = kernelResult.get_expected_image(
                runInstance, displayableSample, cache_kernels2d=Preferences.get().cacheKernels2D
            )
        else:
            sampleImageArr = None
            expectedImageArr = None

        self.widget.updateExpectedImage(
            expectedImageArr,
            autoRange=initialDisplayOfData or self.model.panZoomAutoReset,
            autoLevel=initialDisplayOfData or self.model.autoLevelAutoPerform,
            autoLevelLowerCutoff=self.model.autoLevelLowerCutoff
        )

        self._sampleImageArr = sampleImageArr
        self._expectedImageArr = expectedImageArr

    @pyqtSlot()
    def _uiClickLoadImage(self) -> None:
        """ Loads a sample image to simulate the expected version of. """

        if self.widget.outputDirector() is None:
            return

        displayableSample, okClicked = SampleStructurePickerDialog.getDisplayableSampleForOutput(
            self.widget,
            self.widget.outputDirector().value().results.run_instance.sample_properties.structure
        )

        if okClicked:
            self.widget.outputDirector().setDisplayableSample(displayableSample)

    @pyqtSlot()
    def _uiClickUnloadImage(self) -> None:
        """ Unloads the currently loaded sample image. """

        if self.widget.outputDirector() is None:
            return

        self.widget.outputDirector().clearDisplayableSample()

    @pyqtSlot()
    def _uiClickExportSampleImage(self) -> None:
        """
        Exports the sample image that the displayed expected image is calculated from to a file.
        """

        if self._sampleImageArr is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self.widget,
            caption="Export Sample Image",
            filter="TIFF files (*.tiff)",
            directory=UserFileDirs.SimulatedData
        )

        if path:  # Check whether a file was picked
            imsave(path, self._sampleImageArr.astype(np.float32))

    @pyqtSlot()
    def _uiClickExportExpectedImage(self) -> None:
        """ Exports the currently displayed expected image to a file. """

        if self._expectedImageArr is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self.widget,
            caption="Export Expected Image",
            filter="TIFF files (*.tiff)",
            directory=UserFileDirs.SimulatedData
        )

        if path:  # Check whether a file was picked
            imsave(path, self._expectedImageArr.astype(np.float32))

    @pyqtSlot()
    def _uiClickPanZoomReset(self) -> None:
        self.widget.resetPanZoom()

    @pyqtSlot(int)
    def _uiPanZoomAutoResetChange(self, enabled: bool) -> None:
        self.model.panZoomAutoReset = enabled

    @pyqtSlot()
    def _uiClickAutoLevel(self) -> None:
        self.widget.autoLevel(self.model.autoLevelLowerCutoff)

    @pyqtSlot(int)
    def _uiAutoLevelAutoPerformChange(self, enabled: bool) -> None:
        self.model.autoLevelAutoPerform = enabled

    @pyqtSlot(float)
    def _uiAutoLevelLowerCutoffChange(self, value: float) -> None:
        self.model.autoLevelLowerCutoff = value
