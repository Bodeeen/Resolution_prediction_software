from typing import Optional, List, Union

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal, Qt

from frcpredict.model import FrcCurve, FrcSimulationResults, FrcSimulationResultsView
from frcpredict.ui import BaseWidget
from .frc_results_p import FrcResultsPresenter
from .multivalues_edit import MultivalueListSignals


class FrcResultsWidget(BaseWidget):
    """
    A widget that displays results of an FRC simulation.
    """

    # Signals
    thresholdChanged = pyqtSignal(float)
    importResultsClicked = pyqtSignal()
    exportResultsClicked = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        self._currentFrcCurve = None
        self._thresholdPlotItems = []

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

        # Connect forwarded signals
        self.editThreshold.valueChanged.connect(self.thresholdChanged)
        self.btnImportResults.clicked.connect(self.importResultsClicked)
        self.btnExportResults.clicked.connect(self.exportResultsClicked)

        # Initialize presenter
        self._presenter = FrcResultsPresenter(self)

    def value(self) -> FrcSimulationResultsView:
        return self._presenter.model

    def setValue(self, value: Union[FrcSimulationResults, FrcSimulationResultsView]) -> None:
        self._presenter.model = value

    def updateMultivaluesEditWidgets(self,
                                     results: Optional[FrcSimulationResults]) -> MultivalueListSignals:
        """
        Updates the multivalue editor to contain inputs for each multivalue path used in the
        simulation. Returns an object that contains state change signals, one for each multivalue
        path. The indices in the returned lists of signals correspond to the indices in the
        multivalue path list.
        """

        return self.multivaluesEdit.updateEditWidgets(results)

    def updateDisplayedFrcCurve(self, curve: Optional[FrcCurve],
                                multivalueIndices: List[int], inspectedIndex: int) -> None:
        """ Draws the FRC curve in the FRC plot, and updates the parameter multivalue labels. """

        self._currentFrcCurve = curve
        self.plotFrc.clear()

        if curve is not None:
            self.plotFrc.plot(curve.x, curve.y)
            self.multivaluesEdit.updateMultivalueValues(
                multivalueIndices, curve.multivalue_values, inspectedIndex
            )

        self.editThreshold.setEnabled(curve is not None)
        self.btnExportResults.setEnabled(curve is not None)

    def updateViewOptions(self, threshold: float, inspectedIndex: int,
                          inspectionCurveX: Optional[np.ndarray] = None,
                          inspectionCurveY: Optional[np.ndarray] = None,
                          inspectionLabel: str = "") -> None:
        """
        Updates the threshold as well as the inspection state/plot. If inspectedIndex is zero or
        greater, inspectionCurveX and inspectionCurveY must also be passed.
        """

        if inspectedIndex > -1 and (inspectionCurveX is None or inspectionCurveY is None):
            raise ValueError(
                "inspectedIndex > -1, but inspectionCurveX and/or inspectionCurveY were not given."
            )

        self._updateInspection(inspectedIndex, inspectionCurveX, inspectionCurveY, inspectionLabel)
        self._updateThreshold(threshold, inspectedIndex)

    # Internal methods
    def _updateInspection(self, inspectedIndex: int, inspectionCurveX: Optional[np.ndarray] = None,
                          inspectionCurveY: Optional[np.ndarray] = None, label: str = "") -> None:
        """ Updates the inspection state/plot. """

        # Update multivalue editor
        self.multivaluesEdit.updateInspection(inspectedIndex)

        # Update plots
        self.plotInspection.clear()

        if inspectedIndex > -1:
            if inspectionCurveX is not None and inspectionCurveY is not None:
                self.plotInspection.plot(inspectionCurveX, inspectionCurveY)

            self.plotInspection.setLabel("bottom", label)

            self.plotFrc.setVisible(False)
            self.plotInspection.setVisible(True)
        else:
            self.plotInspection.setVisible(False)
            self.plotFrc.setVisible(True)

    def _updateThreshold(self, threshold: float, inspectedIndex: int) -> None:
        """ Updates the threshold-related values and draws threshold lines in the FRC plot. """

        self.editThreshold.setValue(threshold)

        for plotItem in self._thresholdPlotItems:
            self.plotFrc.removeItem(plotItem)

        self._thresholdPlotItems = []

        if self._currentFrcCurve is not None and inspectedIndex < 0:
            value_at_threshold = self._currentFrcCurve.resolution_at_threshold(threshold)

            # Set resolution value label text
            self.lblResolutionValue.setText(
                ("%.2f nm" % value_at_threshold) if value_at_threshold is not None else ""
            )

            # Draw threshold lines in the FRC plot
            if value_at_threshold is not None:
                self._thresholdPlotItems.append(
                    self.plotFrc.plot([0, 1 / 25], [threshold, threshold],
                                      pen=pg.mkPen("r", style=Qt.DashLine))
                )

                if value_at_threshold != 0:
                    self._thresholdPlotItems.append(
                        self.plotFrc.plot([1 / value_at_threshold, 1 / value_at_threshold], [0, 1],
                                          pen=pg.mkPen("r", style=Qt.DashLine))
                    )
        else:
            self.lblResolutionValue.setText("–")
