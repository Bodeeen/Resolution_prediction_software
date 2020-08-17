from typing import Optional, Union, Tuple, List

import numpy as np
from PyQt5.QtCore import pyqtSignal

from frcpredict.model import  RunInstance, KernelSimulationResult, SimulationResults
from frcpredict.ui import BaseWidget
from .multivalues_edit import MultivalueListSignals
from .output_director_m import SimulationResultsView, ViewOptions
from .output_director_p import OutputDirectorPresenter


class OutputDirectorWidget(BaseWidget):
    """
    A widget that displays results of an FRC simulation.
    """

    # Signals
    viewOptionsChanged = pyqtSignal(ViewOptions)
    kernelResultChanged = pyqtSignal(object, object, bool)
    expectedImageChanged = pyqtSignal(object, object, bool)

    sampleImageChanged = pyqtSignal(object, object)
    thresholdChanged = pyqtSignal(float)
    optimizeClicked = pyqtSignal()
    importResultsClicked = pyqtSignal()
    exportResultsClicked = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:

        super().__init__(__file__, *args, **kwargs)

        # Connect forwarded signals
        self.multivaluesEdit.optimizeClicked.connect(self.optimizeClicked)
        self.btnImportResults.clicked.connect(self.importResultsClicked)
        self.btnExportResults.clicked.connect(self.exportResultsClicked)

        # Initialize presenter
        self._presenter = OutputDirectorPresenter(self)

    def setThreshold(self, threshold: float) -> None:
        self.thresholdChanged.emit(threshold)

    def setSampleImage(self, image: np.ndarray, imageId: str) -> None:
        self.sampleImageChanged.emit(image, imageId)

    def clearSampleImage(self) -> None:
        self.sampleImageChanged.emit(None, None)

    def value(self) -> SimulationResultsView:
        return self._presenter.model

    def setValue(self, value: Union[SimulationResults, SimulationResultsView]) -> None:
        self._presenter.model = value

    def updateMultivaluesEditWidgets(self,
                                     results: Optional[SimulationResults]) -> MultivalueListSignals:
        """
        Updates the multivalue editor to contain inputs for each multivalue path used in the
        simulation. Returns an object that contains state change signals, one for each multivalue
        path. The indices in the returned lists of signals correspond to the indices in the
        multivalue path list.
        """

        return self.multivaluesEdit.updateEditWidgets(results)

    def updateDisplayedKernelResult(self, runInstance: Optional[RunInstance],
                                    kernelResult: Optional[KernelSimulationResult],
                                    multivalueIndices: List[int], inspectedIndex: int,
                                    initialDisplayOfData: bool = False) -> None:
        """
        Updates the parameter multivalue labels, and emits signals to inform that the displayed
        kernel result has been updated.
        """

        if kernelResult is not None:
            self.multivaluesEdit.updateMultivalueValues(multivalueIndices,
                                                        kernelResult.multivalue_values)

        self.btnExportResults.setEnabled(kernelResult is not None)

        # Emit signals
        self.kernelResultChanged.emit(runInstance, kernelResult, initialDisplayOfData)
        self.expectedImageChanged.emit(runInstance, kernelResult, initialDisplayOfData)

    def updateViewOptions(self, viewOptions: ViewOptions) -> None:
        """
        Updates the threshold and inspection state. If inspectedIndex is zero or greater,
        inspectionCurveX and inspectionCurveY must also be passed.
        """

        if viewOptions.inspectedMultivalueIndex > -1 and viewOptions.inspectionDetails is None:
            raise ValueError("inspectedIndex > -1, but inspectionDetails was not set.")

        # Update inspection
        self.multivaluesEdit.updateInspection(viewOptions.inspectedMultivalueIndex)

        # Emit signal
        self.viewOptionsChanged.emit(viewOptions)
