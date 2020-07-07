from functools import reduce
from typing import Optional, List, Union

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal, Qt, pyqtBoundSignal
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSlider

from frcpredict.model import FrcCurve, FrcSimulationResults, FrcSimulationResultsView
from frcpredict.ui import BaseWidget
from frcpredict.util import get_value_from_path
from .frc_results_p import FrcResultsPresenter


class FrcResultsWidget(BaseWidget):
    """
    A widget that displays results of an FRC simulation.
    """

    thresholdChanged = pyqtSignal(float)
    importResultsClicked = pyqtSignal()
    exportResultsClicked = pyqtSignal()

    def __init__(self, *args, **kwargs) -> None:
        self._valueLabels = []
        self._thresholdPlotItem = None

        super().__init__(__file__, *args, **kwargs)

        # Prepare UI elements
        self.plotFrc.setXRange(0, 1 / 25, padding=0)
        self.plotFrc.setYRange(0, 1)

        self.plotFrc.getAxis("bottom").setTicks(
            [[(1 / value, str(value)) for value in [300, 100, 60, 40, 30]]]
        )
        self.plotFrc.setLabel("bottom", "Resolution [nm]")
        self.plotFrc.getAxis("bottom").setHeight(30)

        self.editThreshold.valueChanged.connect(self.thresholdChanged)
        self.btnImportResults.clicked.connect(self.importResultsClicked)
        self.btnExportResults.clicked.connect(self.exportResultsClicked)

        # Initialize presenter
        self._presenter = FrcResultsPresenter(self)

    def value(self) -> FrcSimulationResultsView:
        return self._presenter.model

    def setValue(self, value: Union[FrcSimulationResults, FrcSimulationResultsView]) -> None:
        self._presenter.model = value

    def updateData(self, curve: Optional[FrcCurve]) -> None:
        """ Draws the FRC curve in the FRC plot, and updates the parameter range value labels. """

        self.plotFrc.clear()

        if curve is not None:
            # Update plot
            self.plotFrc.plot(curve.x, curve.y, clickable=True)

            # Update parameter range value labels
            for i in range(0, len(self._valueLabels)):
                self._valueLabels[i].setText("%#.4g" % curve.range_values[i])
        else:
            self.lblResolutionValue.setText("")

        self.editThreshold.setEnabled(curve is not None)
        self.btnExportResults.setEnabled(curve is not None)

    def updateRangePaths(self, results: Optional[FrcSimulationResults]) -> List[pyqtBoundSignal]:
        """
        Updates the range value inputs to contain sliders for each range path used in the
        simulation. Returns a list of value change signals, one for each range path. The indices in
        the returned list of signals correspond to the indices in the range path list.
        """

        sliderValueChangeEvents = []
        valueLabels = []
        
        parameterControlLayout = self.frmRangeParameters.layout()

        # Remove all existing range parameter widgets
        for i in reversed(range(parameterControlLayout.rowCount())):
            parameterControlLayout.removeRow(i)

        # Add new ones
        if results is not None:
            for rangePath in results.range_paths:
                # Field label
                fieldName = reduce(
                    lambda x, y: x + (f"[{y}]" if isinstance(y, int) else f" → {y}"),
                    rangePath
                )

                fieldText = f"{fieldName}:"

                fieldLabelFont = self.font()
                while (QFontMetrics(fieldLabelFont).boundingRect(fieldText).width() >
                       self.frmRangeParameters.width()):  # Adjust font size to fit form
                    fieldLabelFont.setPointSize(fieldLabelFont.pointSize() - 1)

                # Slider
                valueRange = get_value_from_path(results.run_instance, rangePath)

                slider = QSlider(
                    Qt.Horizontal,
                    minimum=0,
                    maximum=valueRange.num_evaluations - 1,
                    value=0
                )

                # Value label
                valueLabel = QLabel(str(valueRange.start))
                valueLabel.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
                valueLabels.append(valueLabel)

                # Layout containing slider and value label
                hBox = QHBoxLayout()
                hBox.addWidget(slider)
                hBox.addWidget(valueLabel)
                hBox.setSpacing(9)
                hBox.setStretch(0, 3)
                hBox.setStretch(1, 2)

                # Append
                parameterControlLayout.addRow(
                    QLabel(fieldText, font=fieldLabelFont),
                    hBox
                )

                sliderValueChangeEvents.append(slider.valueChanged)

        self._valueLabels = valueLabels

        # Return slider value change signals
        return sliderValueChangeEvents

    def updateThreshold(self, curve: Optional[FrcCurve], threshold: float) -> None:
        """ Draws the threshold line in the FRC plot. """

        self.editThreshold.setValue(threshold)

        if self._thresholdPlotItem is not None:
            self.plotFrc.removeItem(self._thresholdPlotItem)

        if curve is not None:
            self._thresholdPlotItem = self.plotFrc.plot([0, 1 / 25], [threshold, threshold],
                                                        pen=pg.mkPen("r", style=Qt.DashLine))

            value_at_threshold = curve.resolution_at_threshold(threshold)
            self.lblResolutionValue.setText(
                ("%.2f nm" % value_at_threshold) if value_at_threshold is not None else ""
            )
        else:
            self.lblResolutionValue.setText("")