from typing import Optional, Tuple

import numpy as np
import pyqtgraph as pg
from pyqtgraph.GraphicsScene import exportDialog
from PyQt5.QtCore import pyqtSignal, Qt

from frcpredict.ui.util import centerWindow, fixPyQtGraphNanBehaviour
from ..controls.output_director_m import ViewOptions, InspectionDetails
from ..base_output_receiver_widget import BaseOutputReceiverWidget
from .frc_results_m import Plot
from .frc_results_p import FrcResultsPresenter


class FrcResultsWidget(BaseOutputReceiverWidget):
    """
    A widget that displays results of an FRC simulation.
    """

    # Signals
    thresholdChanged = pyqtSignal(float)
    crosshairToggled = pyqtSignal(int)

    exportValuesClicked = pyqtSignal()
    exportGraphClicked = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        self._frcThresholdPlotItems = []
        self._inspectionPlotItems = []

        super().__init__(__file__, *args, **kwargs)

        self.exportDialogFrc = exportDialog.ExportDialog(self.plotFrc.scene())
        self.exportDialogInspection = exportDialog.ExportDialog(self.plotInspection.scene())

        # Prepare UI elements
        self.plotFrc.setRange(xRange=[0, 1 / 25], yRange=[-0.1, 1.1], padding=0)
        self.plotFrc.getAxis("bottom").setTicks(
            [[(1 / value, str(value)) for value in [300, 100, 60, 40, 30]]]
        )
        self.plotFrc.setLabel("bottom", "Resolution [nm]")
        self.plotFrc.getAxis("bottom").setHeight(30)
        self.plotFrc.setLabel("left", "FRC")
        self.plotFrc.getAxis("left").setWidth(45)

        self.plotInspection.getAxis("bottom").setHeight(30)
        self.plotInspection.setLabel("left", "Resolution [nm]")
        self.plotInspection.getAxis("left").setWidth(45)

        self.plotFrc.setVisible(True)
        self.plotInspection.setVisible(False)

        # Connect forwarded events
        self.editThreshold.valueChanged.connect(self.thresholdChanged)
        self.chkCrosshair.stateChanged.connect(self.crosshairToggled)

        self.btnExportValues.clicked.connect(self.exportValuesClicked)
        self.btnExportGraph.clicked.connect(self.exportGraphClicked)

        # Initialize presenter
        self._presenter = FrcResultsPresenter(self)

    def showExportGraphDialog(self, plot: Plot) -> None:
        """ Opens the export dialog for the given plot, so that the user may export its visuals. """
        if plot == Plot.frc:
            self.exportDialogFrc.show(self.plotFrc)
            centerWindow(self.exportDialogFrc)
        elif plot == Plot.inspection:
            self.exportDialogInspection.show(self.plotInspection)
            centerWindow(self.exportDialogInspection)
        else:
            raise ValueError("Invalid plot provided")

    def setVisiblePlot(self, plot: Plot) -> None:
        """ Sets which plot should be visible to the user. """
        if plot == plot.frc:
            self.plotInspection.setVisible(False)
            self.plotFrc.setVisible(True)
        elif plot == plot.inspection:
            self.plotFrc.setVisible(False)
            self.plotInspection.setVisible(True)
        else:
            raise ValueError("Invalid plot provided")

    def updateFrcCurve(self, frcCurve: Optional[Tuple[np.ndarray, np.ndarray]]) -> None:
        """ Updates the FRC curve in the FRC plot. """

        self.plotFrc.clear()
        if frcCurve is not None:
            self.plotFrc.plot(*frcCurve)

        self.grpData.setEnabled(frcCurve is not None)
        self.btnExportValues.setEnabled(frcCurve is not None)
        self.btnExportGraph.setEnabled(frcCurve is not None)

    def updateViewOptions(self, viewOptions: ViewOptions, showCrosshair: bool) -> None:
        """
        Updates the threshold as well as the inspection plot. If inspectedIndex is zero or greater,
        inspectionCurveX and inspectionCurveY must also be passed.
        """

        self._updateInspectionPlot(viewOptions.inspectedMultivalueIndex, viewOptions.inspectionDetails,
                                   showCrosshair)
        self._updateThreshold(viewOptions.threshold, viewOptions.valueAtThreshold,
                              showCrosshair)

    # Internal methods
    def _updateInspectionPlot(self, inspectedIndex: int, inspectionDetails: InspectionDetails,
                              showCrosshair: bool) -> None:
        """ Updates the inspection plot curve, labels, and threshold indicators. """

        self.plotInspection.clear()

        if inspectedIndex > -1 and inspectionDetails is not None:
            fixPyQtGraphNanBehaviour()

            inspectionCurveX = inspectionDetails.curveX
            inspectionCurveY = inspectionDetails.curveY
            inspectionCurveIndex = inspectionDetails.curveIndex

            if inspectionCurveX is not None and inspectionCurveY is not None:
                # Update inspection plot curve
                self.plotInspection.plot(inspectionCurveX, inspectionCurveY)

                paddingX = (np.nanmax(inspectionCurveX) - np.nanmin(inspectionCurveX)) / 10
                paddingY = (np.nanmax(inspectionCurveY) - np.nanmin(inspectionCurveY)) / 10

                plotXMin = np.nanmin(inspectionCurveX) - paddingX
                plotXMax = np.nanmax(inspectionCurveX) + paddingX
                plotYMin = np.nanmin(inspectionCurveY) - paddingY
                plotYMax = np.nanmax(inspectionCurveY) + paddingY

                try:
                    self.plotInspection.setRange(xRange=[plotXMin, plotXMax],
                                                 yRange=[plotYMin, plotYMax],
                                                 padding=0)
                except:
                    # Probably raised due to nan range, in turn due to non-intersecting threshold
                    pass

                # Update inspection plot crosshair lines
                for plotItem in self._inspectionPlotItems:
                    self.plotInspection.removeItem(plotItem)

                if showCrosshair:
                    self._inspectionPlotItems = []

                    self._inspectionPlotItems.append(
                        self.plotInspection.plot(
                            [plotXMin, plotXMax],
                            [inspectionCurveY[inspectionCurveIndex],
                             inspectionCurveY[inspectionCurveIndex]],
                            pen=pg.mkPen("r", style=Qt.DashLine)
                        )
                    )

                    self._inspectionPlotItems.append(
                        self.plotInspection.plot(
                            [inspectionCurveX[inspectionCurveIndex],
                             inspectionCurveX[inspectionCurveIndex]],
                            [plotYMin, plotYMax],
                            pen=pg.mkPen("r", style=Qt.DashLine)
                        )
                    )

            # Set label
            self.plotInspection.setLabel("bottom", inspectionDetails.label)

    def _updateFrcThreshold(self, threshold: float, valueAtThreshold: float,
                            showCrosshair: bool) -> None:
        """ Updates the threshold indicators in the FRC plot. """

        # Update FRC plot crosshair lines
        for plotItem in self._frcThresholdPlotItems:
            self.plotFrc.removeItem(plotItem)

        self._frcThresholdPlotItems = []

        if showCrosshair:
            self._frcThresholdPlotItems.append(
                self.plotFrc.plot([0, 1 / 25], [threshold, threshold],
                                  pen=pg.mkPen("r", style=Qt.DashLine))
            )

            if valueAtThreshold:
                self._frcThresholdPlotItems.append(
                    self.plotFrc.plot([1 / valueAtThreshold, 1 / valueAtThreshold], [-0.1, 1.1],
                                      pen=pg.mkPen("r", style=Qt.DashLine))
                )

    def _updateThreshold(self, threshold: float, valueAtThreshold: float,
                         showCrosshair: bool) -> None:
        """ Updates the threshold-related values and draws threshold indicators in the FRC plot. """

        # Update FRC plot crosshair lines
        self._updateFrcThreshold(threshold, valueAtThreshold, showCrosshair)

        # Set resolution value label text
        self.editThreshold.setValue(threshold)

        if valueAtThreshold is not None and valueAtThreshold > 0.0:
            self.lblResolutionValue.setText(
                ("%.2f nm" % valueAtThreshold) if valueAtThreshold is not None else ""
            )
        else:
            self.lblResolutionValue.setText("–")
