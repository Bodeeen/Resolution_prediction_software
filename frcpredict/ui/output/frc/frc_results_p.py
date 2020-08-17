import csv
from typing import Optional, Tuple

import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog

from frcpredict.model import RunInstance, KernelSimulationResult
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import UserFileDirs
from ..controls.output_director_m import ViewOptions
from .frc_results_m import FrcResultsModel, Plot


class FrcResultsPresenter(BasePresenter[FrcResultsModel]):
    """
    Presenter for the FRC/resolution results widget.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: FrcResultsModel) -> None:
        # Disconnect old model event handling
        try:
            model.currentPlotChanged.disconnect(self._onCurrentPlotChange)
            model.crosshairToggled.disconnect(self._onCrosshairToggle)
            model.frcCurveChanged.disconnect(self._onFrcCurveChange)
            model.viewOptionsChanged.disconnect(self._onViewOptionsChange)
        except AttributeError:
            pass

        # Set model
        self._model = model

        # Trigger model change event handlers
        self._onCurrentPlotChange(model.currentPlot)
        self._onCrosshairToggle(model.crosshairEnabled)
        self._onFrcCurveChange(model.frcCurve)
        self._onViewOptionsChange(model.viewOptions)

        # Prepare model events
        model.currentPlotChanged.connect(self._onCurrentPlotChange)
        model.crosshairToggled.connect(self._onCrosshairToggle)
        model.frcCurveChanged.connect(self._onFrcCurveChange)
        model.viewOptionsChanged.connect(self._onViewOptionsChange)

    # Methods
    def __init__(self, widget) -> None:
        super().__init__(FrcResultsModel(), widget)

        # Prepare UI events
        widget.kernelResultChanged.connect(self._uiKernelResultChange)
        widget.viewOptionsChanged.connect(self._uiViewOptionsChange)

        widget.thresholdChanged.connect(self._uiThresholdChange)
        widget.crosshairToggled.connect(self._uiCrosshairToggle)

        widget.exportValuesClicked.connect(self._uiClickExportValues)
        widget.exportGraphClicked.connect(self._uiClickExportGraph)

    # Internal methods
    def _exportFrcPlotToCsv(self, path: str) -> None:
        """ Exports the FRC curve X and Y values to a CSV file. """

        curveX, curveY = self.model.frcCurve

        with open(path, "w") as csvFile:
            csvWriter = csv.writer(csvFile, delimiter=",", lineterminator="\n")
            csvWriter.writerow(["frequency", "frc"])
            for i in range(0, len(curveX)):
                csvWriter.writerow([np.format_float_positional(curveX[i], trim="0"),
                                    np.format_float_positional(curveY[i], trim="0")])

    def _exportInspectionPlotToCsv(self, path: str) -> None:
        """ Exports the inspection curve X and Y values to a CSV file. """

        curveX = self.model.viewOptions.inspectionDetails.curveX
        curveY = self.model.viewOptions.inspectionDetails.curveY
        if len(curveX) != len(curveY):
            raise Exception("Inspection curve X and Y arrays are not of the same length")

        label = self.model.viewOptions.inspectionDetails.label
        if not label:
            label = "parameter value"

        with open(path, "w") as csvFile:
            csvWriter = csv.writer(csvFile, delimiter=",", lineterminator="\n")
            csvWriter.writerow([label, "resolution [nm]"])
            for i in range(0, len(curveX)):
                csvWriter.writerow([np.format_float_positional(curveX[i], trim="0"),
                                    np.format_float_positional(curveY[i], trim="0")])

    # Model event handling
    def _onCurrentPlotChange(self, plot: Plot) -> None:
        self.widget.setVisiblePlot(plot)

    def _onCrosshairToggle(self, crosshairEnabled: bool) -> None:
        self.widget.updateViewOptions(
            self.model.viewOptions,
            crosshairEnabled and self.model.frcCurve is not None
        )

    def _onFrcCurveChange(self, frcCurve: Optional[Tuple[np.ndarray, np.ndarray]]) -> None:
        self.widget.updateFrcCurve(frcCurve)

    def _onViewOptionsChange(self, viewOptions: ViewOptions) -> None:
        self.widget.updateViewOptions(
            viewOptions,
            self.model.crosshairEnabled and self.model.frcCurve is not None
        )

    # UI event handling
    @pyqtSlot(object, object, bool)
    def _uiKernelResultChange(self, runInstance: RunInstance,
                              kernelResult: Optional[KernelSimulationResult], _: bool) -> None:
        self.model.frcCurve = kernelResult.get_frc_curve(runInstance)

    @pyqtSlot(ViewOptions)
    def _uiViewOptionsChange(self, viewOptions: ViewOptions) -> None:
        self.model.viewOptions = viewOptions

        if viewOptions.inspectedMultivalueIndex > -1 and viewOptions.inspectionDetails is not None:
            self.model.currentPlot = Plot.inspection
        else:
            self.model.currentPlot = Plot.frc

    @pyqtSlot(float)
    def _uiThresholdChange(self, threshold: float) -> None:
        if self.widget.outputDirector() is None:
            return

        self.widget.outputDirector().setThreshold(threshold)

    @pyqtSlot(int)
    def _uiCrosshairToggle(self, enabled: bool) -> None:
        self.model.crosshairEnabled = enabled

    @pyqtSlot()
    def _uiClickExportValues(self) -> None:
        """ Exports the current plot curve X and Y values to a CSV file picked by the user. """

        if self.model.currentPlot == Plot.frc:
            if self.model.frcCurve is None:
                return

            path, _ = QFileDialog.getSaveFileName(
                self.widget,
                caption="Export FRC Plot Values",
                filter="CSV files (*.csv)",
                directory=UserFileDirs.SimulatedImages
            )

            if path:  # Check whether a file was picked
                self._exportFrcPlotToCsv(path)
        elif self.model.currentPlot == Plot.inspection:
            if (self.model.viewOptions.inspectedMultivalueIndex < 0
                    or self.model.viewOptions.inspectionDetails is None):
                return

            path, _ = QFileDialog.getSaveFileName(
                self.widget,
                caption="Export Inspection Plot Values",
                filter="CSV files (*.csv)",
                directory=UserFileDirs.SimulatedImages
            )

            if path:  # Check whether a file was picked
                self._exportInspectionPlotToCsv(path)

    @pyqtSlot()
    def _uiClickExportGraph(self) -> None:
        """ Exports the currently displayed expected image to a file picked by the user. """
        self.widget.showExportGraphDialog(self.model.currentPlot)
