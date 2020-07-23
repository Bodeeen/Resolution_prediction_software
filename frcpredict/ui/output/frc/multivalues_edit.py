from dataclasses import dataclass
from typing import Optional, List

from PyQt5.QtCore import pyqtSignal, pyqtBoundSignal, Qt
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSlider, QCheckBox, QFrame

from frcpredict.model import FrcSimulationResults
from frcpredict.ui import BaseWidget
from frcpredict.util import get_value_from_path
from .label_utils import getLabelForMultivalue


@dataclass
class MultivalueListSignals:
    """
    Container for state/value change signals for all represented multivalues. The indices in the
    lists of signals correspond to the indices in the multivalue path list.
    """

    inspectionStateChangeSignals: List[pyqtBoundSignal]
    multivalueValueChangeSignals: List[pyqtBoundSignal]


class MultivaluesEditWidget(BaseWidget):
    # Signals
    optimizeClicked = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        self._inspectionCheckBoxes = []
        self._multivalueSliders = []
        self._multivalueValueLabels = []

        self._currentMultivalueValues = None

        super().__init__(__file__, *args, **kwargs)

        # Prepare UI elements
        self.btnOptimize.setVisible(False)

        # Connect forwarded signals
        self.btnOptimize.clicked.connect(self.optimizeClicked)

    def updateEditWidgets(self,
                          results: Optional[FrcSimulationResults]) -> MultivalueListSignals:
        """
        Updates the multivalue editor to contain inputs for each multivalue path used in the
        simulation. Returns an object that contains state change signals, one for each multivalue
        path. The indices in the returned lists of signals correspond to the indices in the
        multivalue path list.
        """

        inspectionStateChangeSignals = []
        sliderValueChangeSignals = []

        inspectionCheckBoxes = []
        sliders = []
        valueLabels = []

        parameterControlLayout = self.frmMultivalues.layout()

        # Remove all existing multivalue parameter widgets
        for i in reversed(range(parameterControlLayout.rowCount())):
            parameterControlLayout.removeRow(i)

        # Add new ones
        if results is not None:
            for multivaluePath in results.multivalue_paths:
                # Field label
                fieldText = f"{getLabelForMultivalue(results, multivaluePath)}:"

                fieldLabelFont = self.font()
                while (QFontMetrics(fieldLabelFont).boundingRect(fieldText).width() >
                       self.frmMultivalues.width()):
                    # Adjust font size to fit form
                    fieldLabelFont.setPointSize(fieldLabelFont.pointSize() - 1)

                # Checkbox for enabling/disabling inspection of the multivalue
                inspectionCheckBox = QCheckBox(toolTip="Inspect parameter")
                inspectionCheckBoxes.append(inspectionCheckBox)
                inspectionStateChangeSignals.append(inspectionCheckBox.stateChanged)

                # Separator
                separator = QFrame(frameShape=QFrame.VLine, frameShadow=QFrame.Sunken)

                # Slider
                multivalue = get_value_from_path(results.run_instance, multivaluePath)

                slider = QSlider(
                    Qt.Horizontal,
                    minimum=0,
                    maximum=multivalue.num_values() - 1,
                    value=0
                )
                sliders.append(slider)
                sliderValueChangeSignals.append(slider.valueChanged)

                # Value label
                valueLabel = QLabel(str(multivalue.as_array()[0]))
                valueLabel.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
                valueLabels.append(valueLabel)

                # Layout containing checkbox, separator, slider, and value label
                hBox = QHBoxLayout()
                hBox.addWidget(inspectionCheckBox)
                hBox.addWidget(separator)
                hBox.addWidget(slider)
                hBox.addWidget(valueLabel)
                hBox.setSpacing(9)
                hBox.setStretch(0, 0)
                hBox.setStretch(1, 0)
                hBox.setStretch(2, 3)
                hBox.setStretch(3, 2)

                # Append
                parameterControlLayout.addRow(
                    QLabel(fieldText, font=fieldLabelFont),
                    hBox
                )

        self.btnOptimize.setVisible(results is not None and len(results.multivalue_paths) > 0)

        self._inspectionCheckBoxes = inspectionCheckBoxes
        self._multivalueSliders = sliders
        self._multivalueValueLabels = valueLabels

        return MultivalueListSignals(
            inspectionStateChangeSignals=inspectionStateChangeSignals,
            multivalueValueChangeSignals=sliderValueChangeSignals
        )

    def updateInspection(self, inspectedIndex: int) -> None:
        """ Updates the inspection state. """

        # Disable checkboxes and sliders of non-inspected multivalues if currently inspecting one
        for index, checkBox in enumerate(self._inspectionCheckBoxes):
            checkBox.setEnabled(inspectedIndex < 0 or inspectedIndex == index)

        for index, slider in enumerate(self._multivalueSliders):
            slider.setEnabled(inspectedIndex < 0 or inspectedIndex != index)

        # Update multivalue value labels
        self._updateMultivalueValueLabels(inspectedIndex)

    def updateMultivalueValues(self, multivalueIndices: List[int], multivalueValues: List[int],
                               inspectedIndex: int) -> None:
        """ Updates the values for all multivalues in the list of multivalues. """

        for index, slider in enumerate(self._multivalueSliders):
            slider.setValue(multivalueIndices[index])

        self._currentMultivalueValues = multivalueValues
        self._updateMultivalueValueLabels(inspectedIndex)

    # Internal methods
    def _updateMultivalueValueLabels(self, inspectedIndex: int) -> None:
        """ Updates the value labels for all multivalues in the list of multivalues. """

        if self._currentMultivalueValues is None:
            return

        for index, label in enumerate(self._multivalueValueLabels):
            if index != inspectedIndex:
                label.setText("%#.4g" % self._currentMultivalueValues[index])
            else:
                label.setText("Inspecting")
