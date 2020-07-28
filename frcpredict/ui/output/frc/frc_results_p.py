import math
from traceback import format_exc
from typing import Optional, Union

import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMessageBox, QFileDialog

from frcpredict.model import FrcCurve, FrcSimulationResults, FrcSimulationResultsView, JsonContainer
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import UserFileDirs
from .label_utils import getLabelForMultivalue


class FrcResultsPresenter(BasePresenter[FrcSimulationResultsView]):
    """
    Presenter for the FRC results widget.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: Union[FrcSimulationResults, FrcSimulationResultsView]) -> None:
        if isinstance(model, FrcSimulationResults):
            self._model.results = model
        else:
            # Disconnect old model event handling
            try:
                self._model.results_changed.disconnect(self._onResultsChange)
                self._model.inspected_multivalue_index_changed.disconnect(self._onInspectedIndexChange)
                self._model.multivalue_value_index_changed.disconnect(self._onMultivalueIndexChange)
                self._model.threshold_changed.disconnect(self._onThresholdChange)
            except AttributeError:
                pass

            # Set model
            self._model = model

            # Trigger model change event handlers (only results change event, since it in turn also
            # triggers the others)
            self._onResultsChange(model.results)

            # Prepare model events
            model.results_changed.connect(self._onResultsChange)
            model.inspected_multivalue_index_changed.connect(self._onInspectedIndexChange)
            model.multivalue_value_index_changed.connect(self._onMultivalueIndexChange)
            model.threshold_changed.connect(self._onThresholdChange)

    # Methods
    def __init__(self, widget) -> None:
        super().__init__(FrcSimulationResultsView(), widget)

        # Prepare UI events
        widget.thresholdChanged.connect(self._uiThresholdChange)
        widget.optimizeClicked.connect(self._uiClickOptimize)
        widget.importResultsClicked.connect(self._uiClickImportResults)
        widget.exportResultsClicked.connect(self._uiClickExportResults)

    # Internal methods
    def _getCurrentCurve(self) -> Optional[FrcCurve]:
        """
        Returns the currently selected FrcCurve, based on the multivalue indices in the model.
        """

        if self.model.results is not None and len(self.model.results.frc_curves) > 0:
            return self.model.results.frc_curves[tuple(self.model.multivalue_value_indices)]
        else:
            return None

    def _updateDataInWidget(self) -> None:
        """ Updates the widget to show the current FRC curve and view options. """

        self.widget.updateDisplayedFrcCurve(
            self._getCurrentCurve(),
            self.model.multivalue_value_indices, self.model.inspected_multivalue_index
        )
        self._updateViewOptionsInWidget()

    def _updateViewOptionsInWidget(self) -> None:
        """
        Updates the widget to show information matching the current view options (threshold and
        inspection).
        """

        threshold = self.model.threshold
        inspectedIndex = self.model.inspected_multivalue_index

        if inspectedIndex > -1:
            numEvaluations = self.model.results.frc_curves.shape[inspectedIndex]
            label = getLabelForMultivalue(self.model.results,
                                          self.model.results.multivalue_paths[inspectedIndex])

            inspectedCurveX = np.zeros(numEvaluations)
            inspectedCurveY = np.zeros(numEvaluations)

            for i in range(0, numEvaluations):
                multivalueValueIndices = list(self.model.multivalue_value_indices)
                multivalueValueIndices[inspectedIndex] = i
                frcCurveOfEvaluation = self.model.results.frc_curves[tuple(multivalueValueIndices)]

                inspectedCurveX[i] = frcCurveOfEvaluation.multivalue_values[inspectedIndex]
                inspectedCurveY[i] = frcCurveOfEvaluation.resolution_at_threshold(
                    self.model.threshold
                )

            self.widget.updateViewOptions(
                threshold, inspectedIndex, inspectedCurveX, inspectedCurveY, label
            )
        else:
            self.widget.updateViewOptions(threshold, inspectedIndex)

    # Model event handling
    def _onResultsChange(self, results: Optional[FrcSimulationResults]) -> None:
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
            self.model._multivalue_value_indices = [0] * results.frc_curves.ndim
        else:
            self.model._multivalue_value_indices = []

        self.model._inspected_multivalue_index = -1

        # Update widget
        self._updateDataInWidget()

    def _onInspectedIndexChange(self, _) -> None:
        self._updateViewOptionsInWidget()

    def _onMultivalueIndexChange(self, _) -> None:
        self._updateDataInWidget()

    def _onThresholdChange(self, _) -> None:
        self._updateViewOptionsInWidget()

    # UI event handling
    def _uiInspectionStateChange(self, index_of_multivalue: int, inspected: bool) -> None:
        self.model.inspected_multivalue_index = index_of_multivalue if inspected else -1

    def _uiMultivalueChange(self, index_of_multivalue: int, index_in_multivalue: int) -> None:
        self.model.set_multivalue_value(index_of_multivalue, index_in_multivalue)

    @pyqtSlot(float)
    def _uiThresholdChange(self, threshold: float) -> None:
        self.model.threshold = threshold

    @pyqtSlot()
    def _uiClickOptimize(self) -> None:
        """
        Sets the multivalues to the values that give the best resolution at the current threshold.
        """

        if self.model.results.frc_curves is None:
            return

        bestMultivalueIndices = self.model.multivalue_value_indices
        bestResolution = math.inf

        curveIterator = np.nditer(self.model.results.frc_curves, flags=["refs_ok", "multi_index"])
        for curve in curveIterator:
            resolutionForCurve = curve.item().resolution_at_threshold(self.model.threshold)
            if resolutionForCurve is not None and resolutionForCurve < bestResolution:
                bestMultivalueIndices = list(curveIterator.multi_index)
                bestResolution = resolutionForCurve

        self.model.multivalue_value_indices = bestMultivalueIndices

    @pyqtSlot()
    def _uiClickImportResults(self) -> None:
        """ Imports previously saved simulation results from a user-picked file. """

        path, _ = QFileDialog.getOpenFileName(
            self.widget,
            caption="Open results file",
            filter="JSON files (*.json)",
            directory=UserFileDirs.SavedResults
        )

        if path:  # Check whether a file was picked
            with open(path, "r") as jsonFile:
                try:
                    json = jsonFile.read()

                    jsonContainer = JsonContainer[
                        FrcSimulationResults
                    ].from_json_with_converted_dicts(
                        json, FrcSimulationResults
                    )

                    for warning in jsonContainer.validate():
                        QMessageBox.warning(self.widget, "Results load warning", warning)

                    self.model = jsonContainer.data
                except Exception as e:
                    print(format_exc())
                    QMessageBox.critical(self.widget, "Results load error", str(e))

    @pyqtSlot()
    def _uiClickExportResults(self) -> None:
        """ Exports the current simulation results to a user-picked file. """

        path, _ = QFileDialog.getSaveFileName(
            self.widget,
            caption="Save results file",
            filter="JSON files (*.json)",
            directory=UserFileDirs.SavedResults
        )

        if path:  # Check whether a file was picked
            with open(path, "w") as jsonFile:
                jsonContainer = JsonContainer[FrcSimulationResults](self.model.results)
                jsonFile.write(jsonContainer.to_json())
