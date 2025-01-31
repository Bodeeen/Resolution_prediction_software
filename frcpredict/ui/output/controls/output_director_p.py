import math
from copy import copy
from typing import Optional, Union, List

import numpy as np
from PyQt5.QtCore import pyqtSlot

from frcpredict.model import SimulationResults
from frcpredict.ui import BasePresenter, Preferences
from frcpredict.ui.util import getLabelForMultivalue
from frcpredict.util import expand_with_multivalues
from .output_director_m import (
    SimulationResultsView, ViewOptions, InspectionDetails, DisplayableSample
)


class OutputDirectorPresenter(BasePresenter[SimulationResultsView]):
    """
    Presenter for the output controls widget.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: Union[SimulationResults, SimulationResultsView]) -> None:
        if isinstance(model, SimulationResults):
            self._model.results = model
        else:
            # Disconnect old model event handling
            try:
                self._model.resultsChanged.disconnect(self._onResultsChange)
                self._model.inspectedMultivalueIndexChanged.disconnect(self._onInspectedIndexChange)
                self._model.multivalueValueIndexChanged.disconnect(self._onMultivalueIndexChange)
                self._model.displayableSampleChanged.disconnect(self._onDisplayableSampleChange)
                self._model.thresholdChanged.disconnect(self._onThresholdChange)
            except AttributeError:
                pass

            # Set model
            self._model = model

            # Trigger model change event handlers (only results change event, since it in turn also
            # triggers the others)
            self._onResultsChange(model.results)

            # Prepare model events
            model.resultsChanged.connect(self._onResultsChange)
            model.inspectedMultivalueIndexChanged.connect(self._onInspectedIndexChange)
            model.multivalueValueIndexChanged.connect(self._onMultivalueIndexChange)
            model.displayableSampleChanged.connect(self._onDisplayableSampleChange)
            model.thresholdChanged.connect(self._onThresholdChange)

    # Methods
    def __init__(self, widget) -> None:
        self._currentRunInstance = None
        self._currentKernelResult = None
        self._clearCachedInspectionCurves()

        super().__init__(SimulationResultsView(), widget)

        # Prepare UI events
        widget.displayableSampleChanged.connect(self._uiDisplayableSampleChange)
        widget.thresholdChanged.connect(self._uiThresholdChange)
        widget.optimizeClicked.connect(self._uiClickOptimize)

    # Internal methods
    def _updateDataInWidget(self, initialDisplayOfData: bool = False) -> None:
        """ Updates the widget to show the current kernel simulation result and view options. """
        if self.model.results is not None and len(self.model.results.kernel_results) > 0:
            self._currentKernelResult = self.model.results.kernel_results[
                tuple(self.model.multivalueValueIndices)
            ]

            self._currentRunInstance = expand_with_multivalues(
                self.model.results.run_instance,
                self.model.results.multivalue_paths,
                self._currentKernelResult.multivalue_values
            )
        else:
            self._currentKernelResult = None
            self._currentRunInstance = None

        self.widget.updateDisplayedKernelResult(
            runInstance=self._currentRunInstance,
            kernelResult=self._currentKernelResult,
            multivalueIndices=self.model.multivalueValueIndices,
            initialDisplayOfData=initialDisplayOfData
        )
        self._updateViewOptionsInWidget()

    def _updateViewOptionsInWidget(self) -> None:
        """
        Updates the widget to show information matching the current view options (threshold and
        inspection).
        """

        # Threshold
        threshold = self.model.threshold
        valueAtThreshold = None
        if self._currentRunInstance is not None and self._currentKernelResult is not None:
            valueAtThreshold = self._currentKernelResult.resolution_at_threshold(
                self._currentRunInstance, threshold
            )

        # Inspection
        inspectedIndex = self.model.inspectedMultivalueIndex
        if inspectedIndex > -1:
            numEvaluations = self.model.results.kernel_results.shape[inspectedIndex]
            label = getLabelForMultivalue(self.model.results,
                                          self.model.results.multivalue_paths[inspectedIndex])

            if (self._cachedInspectionCurveX is not None and
                    self._cachedInspectionCurveY is not None):
                # Use cached curve
                inspectedCurveX = self._cachedInspectionCurveX
                inspectedCurveY = self._cachedInspectionCurveY
            else:
                inspectedCurveX = np.zeros(numEvaluations)
                inspectedCurveY = np.zeros(numEvaluations)

                for i in range(0, numEvaluations):
                    multivalueValueIndices = list(self.model.multivalueValueIndices)
                    multivalueValueIndices[inspectedIndex] = i
                    kernelResult = self.model.results.kernel_results[tuple(multivalueValueIndices)]

                    inspectedCurveX[i] = kernelResult.multivalue_values[inspectedIndex]
                    inspectedCurveY[i] = kernelResult.resolution_at_threshold(
                        expand_with_multivalues(
                            self.model.results.run_instance,
                            self.model.results.multivalue_paths,
                            kernelResult.multivalue_values
                        ),
                        self.model.threshold
                    )

                self._cachedInspectionCurveX = inspectedCurveX
                self._cachedInspectionCurveY = inspectedCurveY

            self._cachedInspectionMultivalueIndices = copy(self.model.multivalueValueIndices)

            self.widget.updateViewOptions(
                ViewOptions(
                    threshold=threshold, valueAtThreshold=valueAtThreshold,
                    inspectedMultivalueIndex=inspectedIndex, inspectionDetails=InspectionDetails(
                        curveX=inspectedCurveX, curveY=inspectedCurveY,
                        curveIndex=self.model.multivalueValueIndices[inspectedIndex],
                        label=label
                    )
                )
            )
        else:
            self.widget.updateViewOptions(
                ViewOptions(threshold=threshold, valueAtThreshold=valueAtThreshold,
                            inspectedMultivalueIndex=inspectedIndex)
            )

    def _clearCachedInspectionCurves(self) -> None:
        self._cachedInspectionMultivalueIndices = []
        self._cachedInspectionCurveX = None
        self._cachedInspectionCurveY = None

    # Model event handling
    def _onResultsChange(self, results: Optional[SimulationResults]) -> None:
        multivaluesEditSignals = self.widget.updateMultivaluesEditWidgets(
            results
        )

        # Prepare handling of inspection state change events
        for signalIndex, signal in enumerate(multivaluesEditSignals.inspectionStateChangeSignals):
            signal.connect(
                lambda value, index=signalIndex: self._uiInspectionStateChange(index, value)
            )

        # Prepare handling of for multivalue change events
        for signalIndex, signal in enumerate(multivaluesEditSignals.multivalueValueChangeSignals):
            signal.connect(
                lambda value, index=signalIndex: self._uiMultivalueChange(index, value)
            )

        # Reset multivalue and inspection state
        if results is not None:
            self.model._multivalueValueIndices = [0] * results.kernel_results.ndim  # All zeroes
        else:
            self.model._multivalueValueIndices = []

        self.model._inspectedMultivalueIndex = -1

        self._clearCachedInspectionCurves()

        # Update widget
        self._updateDataInWidget(initialDisplayOfData=True)

    def _onDisplayableSampleChange(self, displayableSample: DisplayableSample) -> None:
        if Preferences.get().precacheExpectedImages and self.model.results is not None:
            self.model.results.precache(cache_kernels2d=Preferences.get().cacheKernels2D,
                                        cache_expected_image_for=displayableSample)

        self._updateDataInWidget(initialDisplayOfData=True)

    def _onInspectedIndexChange(self, _) -> None:
        self._clearCachedInspectionCurves()
        self._updateViewOptionsInWidget()

    def _onMultivalueIndexChange(self, multivalueIndices: List[int]) -> None:
        # Clear cache if needed
        if len(multivalueIndices) != len(self._cachedInspectionMultivalueIndices):
            self._clearCachedInspectionCurves()
        else:
            oldIndicesExceptInspected = [
                element
                for i, element in enumerate(multivalueIndices)
                if i != self.model.inspectedMultivalueIndex
            ]

            newIndicesExceptInspected = [
                element
                for i, element in enumerate(self._cachedInspectionMultivalueIndices)
                if i != self.model.inspectedMultivalueIndex
            ]

            if oldIndicesExceptInspected != newIndicesExceptInspected:
                self._clearCachedInspectionCurves()

        # Update data
        self._updateDataInWidget()

    def _onThresholdChange(self, _) -> None:
        self._clearCachedInspectionCurves()
        self._updateViewOptionsInWidget()

    # UI event handling
    def _uiInspectionStateChange(self, index_of_multivalue: int, inspected: bool) -> None:
        self.model.inspectedMultivalueIndex = index_of_multivalue if inspected else -1

    def _uiMultivalueChange(self, index_of_multivalue: int, index_in_multivalue: int) -> None:
        self.model.setMultivalueValue(index_of_multivalue, index_in_multivalue)

    @pyqtSlot(object)
    def _uiDisplayableSampleChange(self, displayableSample: Optional[DisplayableSample]) -> None:
        self.model.displayableSample = displayableSample

    @pyqtSlot(float)
    def _uiThresholdChange(self, threshold: float) -> None:
        self.model.threshold = threshold

    @pyqtSlot()
    def _uiClickOptimize(self) -> None:
        """
        Sets the multivalues to the values that give the best resolution at the current threshold.
        """

        if self.model.results.kernel_results is None:
            return

        bestMultivalueIndices = self.model.multivalueValueIndices
        bestResolution = math.inf

        kernelResultIterator = np.nditer(self.model.results.kernel_results,
                                         flags=["refs_ok", "multi_index"])

        for kernelResult in kernelResultIterator:
            kernelResult = kernelResult.item()

            runInstance = expand_with_multivalues(
                self.model.results.run_instance,
                self.model.results.multivalue_paths,
                kernelResult.multivalue_values
            )

            resolutionForKernelResult = kernelResult.resolution_at_threshold(runInstance,
                                                                             self.model.threshold)

            if resolutionForKernelResult is not None and resolutionForKernelResult < bestResolution:
                bestMultivalueIndices = list(kernelResultIterator.multi_index)
                bestResolution = resolutionForKernelResult

        self.model.multivalueValueIndices = bestMultivalueIndices
