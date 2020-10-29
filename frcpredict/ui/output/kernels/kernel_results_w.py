from typing import Optional

import numpy as np
from PyQt5.QtCore import pyqtSignal

from frcpredict.model import KernelType
from ..base_output_receiver_widget import BaseOutputReceiverWidget
from .kernel_results_p import KernelResultsPresenter


class KernelResultsWidget(BaseOutputReceiverWidget):
    """
    A widget that displays simulated kernels.
    """

    # Signals
    kernelResultChanged = pyqtSignal(object, object, bool)
    expKernelToggled = pyqtSignal(bool)
    varKernelToggled = pyqtSignal(bool)
    exportImageClicked = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        self._kernelType = KernelType.exp_kernel

        super().__init__(__file__, *args, **kwargs)

        # Prepare UI elements
        self.imgKernel.ui.roiBtn.setVisible(False)
        self.imgKernel.ui.menuBtn.setVisible(False)

        # Connect forwarded events
        self.rdoExpKernel.clicked.connect(self.expKernelToggled)
        self.rdoVarKernel.clicked.connect(self.varKernelToggled)
        self.btnExportImage.clicked.connect(self.exportImageClicked)

        # Initialize presenter
        self._presenter = KernelResultsPresenter(self)

    def updateKernelType(self, kernelType: KernelType) -> None:
        """ Updates the selected kernel type in the widget. """
        if kernelType == KernelType.exp_kernel:
            self.rdoExpKernel.setChecked(True)
        elif kernelType == KernelType.var_kernel:
            self.rdoVarKernel.setChecked(True)

    def updateKernelImage(self, kernel: Optional[np.ndarray]) -> None:
        """ Updates the kernel image in the widget. """

        # Update image & border
        self.imgKernel.setImage(kernel if kernel is not None else np.zeros((0, 0)))
        self.imgKernel.getImageItem().setBorder({"color": "FF0", "width": 1})

        # Update controls
        self.imgKernel.setEnabled(kernel is not None)
        self.grpViewOptions.setEnabled(kernel is not None)
        self.btnExportImage.setEnabled(kernel is not None)
