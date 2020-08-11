from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSlot, Qt

from frcpredict.model import RunInstance, KernelSimulationResult
from frcpredict.ui import BaseWidget
from ..controls.output_director_w import OutputDirectorWidget


class FrcResultsWidget(BaseWidget):
    """
    A widget that displays results of an FRC simulation.
    """

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        self._outputDirector = None
        self._thresholdPlotItems = []
        self._inspectionPlotItems = []

        super().__init__(__file__, *args, **kwargs)

        # Prepare UI elements
        self.plotFrc.setXRange(0, 1 / 25, padding=0)
        self.plotFrc.setYRange(0, 1)

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

        # Prepare UI events
        self.editThreshold.valueChanged.connect(self._onThresholdChange)

    def outputDirector(self) -> OutputDirectorWidget:
        """ Returns the output director that this widget will receive its data from. """
        return self._outputDirector

    def setOutputDirector(self, outputDirector: OutputDirectorWidget) -> None:
        """ Sets the output director that this widget will receive its data from. """

        if self._outputDirector is not None:
            self._outputDirector.kernelResultsChanged.disconnect(self._onKernelResultsChange)
            self._outputDirector.viewOptionsChanged.disconnect(self._onViewOptionsChange)

        if outputDirector is not None:
            outputDirector.kernelResultsChanged.connect(self._onKernelResultsChange)
            outputDirector.viewOptionsChanged.connect(self._onViewOptionsChange)

        self._outputDirector = outputDirector

    @pyqtSlot(object, object, bool)
    def _onKernelResultsChange(self, run_instance: RunInstance,
                               kernelResult: Optional[KernelSimulationResult], _: bool) -> None:
        """ Draws the FRC curve in the FRC plot. """

        self.plotFrc.clear()
        if kernelResult is not None:
            self.plotFrc.plot(*kernelResult.get_frc_curve(run_instance))

        self.editThreshold.setEnabled(kernelResult is not None)

    @pyqtSlot(float, float, int, object, object, object, str)
    def _onViewOptionsChange(self, threshold: float, valueAtThreshold: float, inspectedIndex: int,
                             inspectionCurveX: Optional[np.ndarray] = None,
                             inspectionCurveY: Optional[np.ndarray] = None,
                             inspectionCurveIndex: Optional[int] = None,
                             inspectionLabel: str = "") -> None:
        """
        Updates the threshold as well as the inspection plot. If inspectedIndex is zero or greater,
        inspectionCurveX and inspectionCurveY must also be passed.
        """

        if inspectedIndex > -1 and (inspectionCurveX is None or inspectionCurveY is None):
            raise ValueError(
                "inspectedIndex > -1, but inspectionCurveX, inspectionCurveY and/or" +
                " inspectionCurveIndex were not set."
            )

        self._updateInspectionPlot(inspectedIndex, inspectionCurveX, inspectionCurveY,
                                   inspectionCurveIndex, inspectionLabel)
        self._updateThreshold(threshold, valueAtThreshold)

    # Internal methods
    def _updateInspectionPlot(self, inspectedIndex: int,
                              inspectionCurveX: Optional[np.ndarray] = None,
                              inspectionCurveY: Optional[np.ndarray] = None,
                              inspectionCurveIndex: Optional[int] = None, label: str = "") -> None:
        """ Updates the inspection plot. """

        self.plotInspection.clear()

        if inspectedIndex > -1:
            if inspectionCurveX is not None and inspectionCurveY is not None:
                # Update inspection plot curve
                self.plotInspection.plot(inspectionCurveX, inspectionCurveY)

                # Update inspection plot crosshair lines
                for plotItem in self._inspectionPlotItems:
                    self.plotInspection.removeItem(plotItem)

                self._inspectionPlotItems = []

                paddingX = (inspectionCurveX.max() - inspectionCurveX.min()) / 10
                self._inspectionPlotItems.append(
                    self.plotInspection.plot(
                        [inspectionCurveX.min() - paddingX, inspectionCurveX.max() + paddingX],
                        [inspectionCurveY[inspectionCurveIndex],
                         inspectionCurveY[inspectionCurveIndex]],
                        pen=pg.mkPen("r", style=Qt.DashLine)
                    )
                )

                paddingY = (inspectionCurveY.max() - inspectionCurveY.min()) / 10
                self._inspectionPlotItems.append(
                    self.plotInspection.plot(
                        [inspectionCurveX[inspectionCurveIndex],
                         inspectionCurveX[inspectionCurveIndex]],
                        [inspectionCurveY.min() - paddingY, inspectionCurveY.max() + paddingY],
                        pen=pg.mkPen("r", style=Qt.DashLine)
                    )
                )

            # Set label
            self.plotInspection.setLabel("bottom", label)

            # Set visibility
            self.plotFrc.setVisible(False)
            self.plotInspection.setVisible(True)
        else:
            self.plotInspection.setVisible(False)
            self.plotFrc.setVisible(True)

    def _updateThreshold(self, threshold: float, valueAtThreshold: float) -> None:
        """ Updates the threshold-related values and draws threshold lines in the FRC plot. """

        # Update FRC plot crosshair lines
        for plotItem in self._thresholdPlotItems:
            self.plotFrc.removeItem(plotItem)

        self._thresholdPlotItems = []

        self._thresholdPlotItems.append(
            self.plotFrc.plot([0, 1 / 25], [threshold, threshold],
                              pen=pg.mkPen("r", style=Qt.DashLine))
        )

        if valueAtThreshold != 0.0:
            self._thresholdPlotItems.append(
                self.plotFrc.plot([1 / valueAtThreshold, 1 / valueAtThreshold], [0, 1],
                                  pen=pg.mkPen("r", style=Qt.DashLine))
            )

        # Set resolution value label text
        self.editThreshold.setValue(threshold)

        if valueAtThreshold is not None and valueAtThreshold > 0.0:
            self.lblResolutionValue.setText(
                ("%.2f nm" % valueAtThreshold) if valueAtThreshold is not None else ""
            )
        else:
            self.lblResolutionValue.setText("–")

    # Event handling
    @pyqtSlot(float)
    def _onThresholdChange(self, threshold: float) -> None:
        if self.outputDirector() is None:
            return

        self.outputDirector().setThreshold(threshold)
