from traceback import format_exc
from typing import Optional, Union

import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMessageBox, QFileDialog

from frcpredict.model import FrcCurve, FrcSimulationResults, FrcSimulationResultsView, JsonContainer
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import UserFileDirs


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
            # Set model
            self._model = model

            # Trigger model change event handlers
            self._onResultsChange(model.results)

            # Prepare model events
            model.results_changed.connect(self._onResultsChange)
            model.range_value_index_changed.connect(self._onRangeValueIndexChange)
            model.threshold_changed.connect(self._onThresholdChange)

    # Methods
    def __init__(self, widget) -> None:
        # Initialize model
        model = FrcSimulationResultsView(
            results=None,
            range_value_indices=[],
            threshold=0.15
        )

        super().__init__(model, widget)

        # Prepare UI events
        widget.thresholdChanged.connect(self._uiThresholdChange)
        widget.importResultsClicked.connect(self._uiClickImportResults)
        widget.exportResultsClicked.connect(self._uiClickExportResults)

    # Internal methods
    def _getCurrentCurve(self) -> Optional[FrcCurve]:
        """
        Returns the currently selected FrcCurve, based on the range value indices in the model.
        """

        if self.model.results is not None and len(self.model.results.frc_curves) > 0:
            return self.model.results.frc_curves[tuple(self.model.range_value_indices)]
        else:
            return None

    def _updateCurveInWidget(self) -> None:
        """ TODO. """

        currentCurve = self._getCurrentCurve()
        self.widget.updateData(currentCurve)
        self.widget.updateThreshold(currentCurve, self.model.threshold)

    # Model event handling
    def _onResultsChange(self, results: Optional[FrcSimulationResults]) -> None:
        rangeValueChangeSignals = self.widget.updateRangePaths(results)

        # Prepare events for sliders
        for signalIndex, signal in enumerate(rangeValueChangeSignals):
            signal.connect(lambda value, index=signalIndex: self._uiRangeValueChange(index, value))

        # Reset range value indices
        if results is not None:
            self.model._range_value_indices = [0] * results.frc_curves.ndim
        else:
            self.model._range_value_indices = []

        # Update widget
        self._updateCurveInWidget()

    def _onRangeValueIndexChange(self, _) -> None:
        self._updateCurveInWidget()

    def _onThresholdChange(self, threshold: float) -> None:
        self.widget.updateThreshold(self._getCurrentCurve(), threshold)

    # UI event handling
    def _uiRangeValueChange(self, index_of_range: int, index_in_range: int) -> None:
        self.model.set_range_value(index_of_range, index_in_range)

    @pyqtSlot(float)
    def _uiThresholdChange(self, threshold: float) -> None:
        self.model.threshold = threshold

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
