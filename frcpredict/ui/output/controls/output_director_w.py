from typing import Optional, List

from PyQt5.QtCore import pyqtSignal

from frcpredict.model import (
    RunInstance, KernelSimulationResult, SimulationResults, DisplayableSample
)
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

    displayableSampleChanged = pyqtSignal(object)
    thresholdChanged = pyqtSignal(float)
    optimizeClicked = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:

        super().__init__(__file__, *args, **kwargs)

        # Connect forwarded signals
        self.multivaluesEdit.optimizeClicked.connect(self.optimizeClicked)

        # Initialize presenter
        self._presenter = OutputDirectorPresenter(self)

    def precacheAllResults(self) -> None:
        """ Pre-caches all results from the currently loaded simulation. """
        self.value().precacheAllResults()

    def setThreshold(self, threshold: float) -> None:
        """ Sets the FRC threshold to the given value. """
        self.thresholdChanged.emit(threshold)

    def setDisplayableSample(self, displayableSample: DisplayableSample) -> None:
        """ Sets the loaded sample image. """
        self.displayableSampleChanged.emit(displayableSample)

    def clearDisplayableSample(self) -> None:
        """ Unloads any loaded sample image. """
        self.displayableSampleChanged.emit(None)

    def simulationResults(self) -> SimulationResults:
        """ Returns the currently loaded simulation results. """
        return self._presenter.model.results

    def setSimulationResults(self, simulationResults: SimulationResults) -> None:
        """ Loads the given simulation results. """
        self._presenter.model = simulationResults

    def value(self) -> SimulationResultsView:
        return self._presenter.model

    def setValue(self, value: SimulationResultsView) -> None:
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
                                    multivalueIndices: List[int],
                                    initialDisplayOfData: bool = False) -> None:
        """
        Updates the parameter multivalue labels, and emits signals to inform that the displayed
        kernel result has been updated.
        """

        if kernelResult is not None:
            self.multivaluesEdit.updateMultivalueValues(multivalueIndices,
                                                        kernelResult.multivalue_values)

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
