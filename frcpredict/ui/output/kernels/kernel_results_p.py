from typing import Optional

import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog
from skimage.io import imsave

from frcpredict.model import RunInstance, KernelType, KernelSimulationResult
from frcpredict.ui import BasePresenter, Preferences
from frcpredict.ui.util import UserFileDirs
from .kernel_results_m import KernelResultsModel


class KernelResultsPresenter(BasePresenter[KernelResultsModel]):
    """
    Presenter for the kernel results widget.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: KernelResultsModel) -> None:
        # Disconnect old model event handling
        try:
            model.kernelTypeChanged.disconnect(self._onKernelTypeChange)
            model.kernels2DChanged.disconnect(self._onKernels2DChange)
        except AttributeError:
            pass

        # Set model
        self._model = model

        # Trigger model change event handlers
        self._onKernelTypeChange(model.kernelType)

        # Prepare model events
        model.kernelTypeChanged.connect(self._onKernelTypeChange)
        model.kernels2DChanged.connect(self._onKernels2DChange)

    # Methods
    def __init__(self, widget) -> None:
        super().__init__(KernelResultsModel(), widget)

        # Prepare UI events
        widget.kernelResultChanged.connect(self._uiKernelResultChange)
        widget.expKernelToggled.connect(self._uiToggleExpKernel)
        widget.varKernelToggled.connect(self._uiToggleVarKernel)
        widget.exportImageClicked.connect(self._uiClickExportImage)

    # Internal methods
    def _getCurrentKernelImage(self) -> Optional[np.ndarray]:
        if self.model.kernels2D is not None:
            return self.model.kernels2D[self.model.kernelType.value]
        else:
            return None

    # Model event handling
    def _onKernelTypeChange(self, kernelType: KernelType) -> None:
        self.widget.updateKernelType(kernelType)
        self.widget.updateKernelImage(self._getCurrentKernelImage())

    def _onKernels2DChange(self, _) -> None:
        self.widget.updateKernelImage(self._getCurrentKernelImage())

    # UI event handling
    @pyqtSlot(object, object, bool)
    def _uiKernelResultChange(self, runInstance: RunInstance,
                              kernelResult: Optional[KernelSimulationResult], _) -> None:
        self.model.kernels2D = kernelResult.get_kernels2d(runInstance,
                                                          cache=Preferences.get().cacheKernels2D)

    @pyqtSlot(bool)
    def _uiToggleExpKernel(self, enabled: bool) -> None:
        if enabled:
            self.model.kernelType = KernelType.exp_kernel

    @pyqtSlot(bool)
    def _uiToggleVarKernel(self, enabled: bool) -> None:
        if enabled:
            self.model.kernelType = KernelType.var_kernel

    @pyqtSlot()
    def _uiClickExportImage(self) -> None:
        """ Exports the currently displayed kernel image to a file picked by the user. """

        caption = ("Export Emission Kernel Image" if self.model.kernelType == KernelType.exp_kernel
                   else "Export Variance Kernel Image")

        path, _ = QFileDialog.getSaveFileName(
            self.widget,
            caption=caption,
            filter="TIFF files (*.tiff)",
            directory=UserFileDirs.SimulatedData
        )

        if path:  # Check whether a file was picked
            imsave(path, self._getCurrentKernelImage().astype(np.float32))
