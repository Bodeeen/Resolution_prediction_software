import random
import string
from typing import Optional

import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog
from skimage.io import imsave

from frcpredict.model import RunInstance, KernelSimulationResult
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import UserFileDirs
from .sample_image_picker_dialog_w import SampleImagePickerDialog
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
        self._expectedImageArr = None  # TODO: Should be in model but initialDisplayOfData prevents it
        super().__init__(VirtualImagingResultsModel(), widget)

        # Prepare UI events
        widget.expectedImageChanged.connect(self._uiExpectedImageChange)

        widget.loadImageClicked.connect(self._uiClickLoadImage)
        widget.unloadImageClicked.connect(self._uiClickUnloadImage)
        widget.exportImageClicked.connect(self._uiClickExportImage)

        widget.panZoomResetClicked.connect(self._uiClickPanZoomReset)
        widget.panZoomAutoResetChanged.connect(self._uiPanZoomAutoResetChange)

        widget.autoLevelClicked.connect(self._uiClickAutoLevel)
        widget.autoLevelAutoPerformChanged.connect(self._uiAutoLevelAutoPerformChange)
        widget.autoLevelLowerCutoffChanged.connect(self._uiAutoLevelLowerCutoffChange)

    # UI event handling
    @pyqtSlot(object, object, bool)
    def _uiExpectedImageChange(self, runInstance: RunInstance,
                               kernelResult: Optional[KernelSimulationResult],
                               initialDisplayOfData: bool) -> None:
        """
        Retrieves the expected image for the given kernel simulation result and tells the widget to
        display it.
        """

        if self.widget.outputDirector() is None:
            return

        sampleImage = self.widget.outputDirector().value().sampleImage

        if kernelResult is not None and sampleImage is not None:
            sampleImageId = sampleImage.id
            sampleImageArr = sampleImage.imageArr
            expectedImageArr = kernelResult.get_expected_image(runInstance, sampleImageId, sampleImageArr)
        else:
            expectedImageArr = None

        self.widget.updateExpectedImage(
            expectedImageArr,
            autoRange=initialDisplayOfData or self.model.panZoomAutoReset,
            autoLevel=initialDisplayOfData or self.model.autoLevelAutoPerform,
            autoLevelLowerCutoff=self.model.autoLevelLowerCutoff
        )
        self._expectedImageArr = expectedImageArr

    @pyqtSlot()
    def _uiClickLoadImage(self) -> None:
        """ Loads a sample image to simulate the expected version of. """

        if self.widget.outputDirector() is None:
            return

        imageData, okClicked = SampleImagePickerDialog.getImageData(
            self.widget,
            self.widget.outputDirector().value().results.run_instance.sample_properties.loaded_structure_id
        )

        if okClicked:
            imageArr = imageData.image.get_numpy_array()

            if not imageData.fromFile:
                imageId = imageData.sampleStructureId
            else:
                # Generate random ID for caching purposes
                imageId = "".join(random.choices(string.ascii_letters + string.digits, k=32))

            self.widget.outputDirector().setSampleImage(imageArr, imageId)

    @pyqtSlot()
    def _uiClickUnloadImage(self) -> None:
        """ Unloads the currently loaded sample image. """

        if self.widget.outputDirector() is None:
            return

        self.widget.outputDirector().clearSampleImage()

    @pyqtSlot()
    def _uiClickExportImage(self) -> None:
        """ Exports the currently displayed expected image to a file. """

        if self._expectedImageArr is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self.widget,
            caption="Export expected image",
            filter="TIFF files (*.tiff)",
            directory=UserFileDirs.SimulatedImages
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
