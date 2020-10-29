from typing import Optional

import numpy as np
from PyQt5.QtCore import pyqtSignal
from pyqtgraph.GraphicsScene import exportDialog

from frcpredict.model import KernelType
from frcpredict.ui.util import centerWindow
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
    exportVisualizationClicked = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        self._kernelType = KernelType.exp_kernel

        super().__init__(__file__, *args, **kwargs)

        self.exportDialogImageView = exportDialog.ExportDialog(self.imgKernel.scene)

        # Prepare UI elements
        self.imgKernel.ui.roiBtn.setVisible(False)
        self.imgKernel.ui.menuBtn.setVisible(False)
        self.imgKernel.getImageItem().setBorder({"color": "FF0", "width": 1})

        # Connect forwarded events
        self.rdoExpKernel.clicked.connect(self.expKernelToggled)
        self.rdoVarKernel.clicked.connect(self.varKernelToggled)
        self.btnExportImage.clicked.connect(self.exportImageClicked)
        self.btnExportVisualization.clicked.connect(self.exportVisualizationClicked)

        # Initialize presenter
        self._presenter = KernelResultsPresenter(self)

    def showExportVisualizationDialog(self) -> None:
        """ Opens the export dialog for the ImageView, so that the user may export its visuals. """
        self.exportDialogImageView.show(self.imgKernel.getImageItem())
        centerWindow(self.exportDialogImageView)

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

        # Update controls
        self.imgKernel.setEnabled(kernel is not None)
        self.grpViewOptions.setEnabled(kernel is not None)
        self.btnExportImage.setEnabled(kernel is not None)
        self.btnExportVisualization.setEnabled(kernel is not None)
