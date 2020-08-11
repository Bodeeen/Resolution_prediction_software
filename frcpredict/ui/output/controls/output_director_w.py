from typing import Optional, Union, Tuple, List

import numpy as np
from PyQt5.QtCore import pyqtSignal

from frcpredict.model import  RunInstance, KernelSimulationResult, SimulationResults
from frcpredict.ui import BaseWidget
from .multivalues_edit import MultivalueListSignals
from .output_director_m import SimulationResultsView
from .output_director_p import OutputDirectorPresenter


class OutputDirectorWidget(BaseWidget):
    """
    A widget that displays results of an FRC simulation.
    """

    # Signals
    viewOptionsChanged = pyqtSignal(float, float, int, object, object, str)
    kernelResultsChanged = pyqtSignal(object, object, bool)
    expectedImageChanged = pyqtSignal(object, object, bool)

    sampleImageChanged = pyqtSignal(object, object)
    thresholdChanged = pyqtSignal(float)
    optimizeClicked = pyqtSignal()
    importResultsClicked = pyqtSignal()
    exportResultsClicked = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        self._currentRunInstance = None
        self._currentKernelResults = None

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

        self._currentRunInstance = runInstance
        self._currentKernelResults = kernelResult

        if kernelResult is not None:
            self.multivaluesEdit.updateMultivalueValues(
                multivalueIndices, kernelResult.multivalue_values, inspectedIndex
            )

        self.btnExportResults.setEnabled(kernelResult is not None)

        # Emit signals
        self.kernelResultsChanged.emit(runInstance, kernelResult, initialDisplayOfData)
        self.expectedImageChanged.emit(runInstance, kernelResult, initialDisplayOfData)

    def updateViewOptions(self, threshold: float, inspectedIndex: int,
                          inspectionCurveX: Optional[np.ndarray] = None,
                          inspectionCurveY: Optional[np.ndarray] = None,
                          inspectionLabel: str = "") -> None:
        """
        Updates the threshold and inspection state. If inspectedIndex is zero or greater,
        inspectionCurveX and inspectionCurveY must also be passed.
        """

        if inspectedIndex > -1 and (inspectionCurveX is None or inspectionCurveY is None):
            raise ValueError(
                "inspectedIndex > -1, but inspectionCurveX and/or inspectionCurveY were not given."
            )

        # Update inspection
        self.multivaluesEdit.updateInspection(inspectedIndex)

        # Update threshold
        valueAtThreshold = None
        if (self._currentRunInstance is not None and self._currentKernelResults is not None
                and inspectedIndex < 0):
            valueAtThreshold = self._currentKernelResults.resolution_at_threshold(
                self._currentRunInstance, threshold
            )

        # Emit signal
        self.viewOptionsChanged.emit(
            threshold, valueAtThreshold if valueAtThreshold is not None else 0.0,
            inspectedIndex, inspectionCurveX, inspectionCurveY, inspectionLabel
        )
