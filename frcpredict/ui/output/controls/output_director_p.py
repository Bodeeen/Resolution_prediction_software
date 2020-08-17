import math
import pickle
from copy import deepcopy
from traceback import format_exc
from typing import Optional, Union

import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMessageBox, QFileDialog

from frcpredict.model import SimulationResults, PersistentContainer
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import UserFileDirs, getLabelForMultivalue
from frcpredict.util import clear_signals, rebuild_dataclass, expand_with_multivalues
from .output_director_m import SimulationResultsView, SampleImage, ViewOptions, InspectionDetails


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
                self._model.sampleImageChanged.disconnect(self._onSampleImageChange)
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
            model.sampleImageChanged.connect(self._onSampleImageChange)
            model.thresholdChanged.connect(self._onThresholdChange)

    # Methods
    def __init__(self, widget) -> None:
        self._currentRunInstance = None
        self._currentKernelResult = None

        super().__init__(SimulationResultsView(), widget)

        # Prepare UI events
        widget.sampleImageChanged.connect(self._uiSampleImageChange)
        widget.thresholdChanged.connect(self._uiThresholdChange)
        widget.optimizeClicked.connect(self._uiClickOptimize)
        widget.importResultsClicked.connect(self._uiClickImportResults)
        widget.exportResultsClicked.connect(self._uiClickExportResults)

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
            inspectedIndex=self.model.inspectedMultivalueIndex,
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

            inspectedCurveX = np.zeros(numEvaluations)
            inspectedCurveY = np.zeros(numEvaluations)

            for i in range(0, numEvaluations):
                multivalueValueIndices = list(self.model.multivalueValueIndices)
                multivalueValueIndices[inspectedIndex] = i
                kernelResult = self.model.results.kernel_results[tuple(multivalueValueIndices)]

                inspectedCurveX[i] = kernelResult.multivalue_values[inspectedIndex]
                inspectedCurveY[i] = kernelResult.resolution_at_threshold(
                    self.model.results.run_instance, self.model.threshold
                )

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
            self.model._multivalueValueIndices = [0] * results.kernel_results.ndim
        else:
            self.model._multivalueValueIndices = []

        self.model._inspectedMultivalueIndex = -1

        # Update widget
        self._updateDataInWidget(initialDisplayOfData=True)

    def _onSampleImageChange(self, _) -> None:
        self._updateDataInWidget(initialDisplayOfData=True)

    def _onInspectedIndexChange(self, _) -> None:
        self._updateViewOptionsInWidget()

    def _onMultivalueIndexChange(self, _) -> None:
        self._updateDataInWidget()

    def _onThresholdChange(self, _) -> None:
        self._updateViewOptionsInWidget()

    # UI event handling
    def _uiInspectionStateChange(self, index_of_multivalue: int, inspected: bool) -> None:
        self.model.inspectedMultivalueIndex = index_of_multivalue if inspected else -1

    def _uiMultivalueChange(self, index_of_multivalue: int, index_in_multivalue: int) -> None:
        self.model.setMultivalueValue(index_of_multivalue, index_in_multivalue)

    @pyqtSlot(object, object)
    def _uiSampleImageChange(self, imageArr: Optional[np.ndarray], imageId: Optional[str]) -> None:
        if imageArr is not None and imageId is not None:
            self.model.sampleImage = SampleImage(id=imageId, imageArr=imageArr)
        elif imageArr is None and imageId is None:
            self.model.sampleImage = None
        else:
            raise ValueError("imageArr and imageId must both either be None or not None")

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
            resolutionForKernelResult = kernelResult.item().resolution_at_threshold(
                self.model.results.run_instance, self.model.threshold
            )

            if resolutionForKernelResult is not None and resolutionForKernelResult < bestResolution:
                bestMultivalueIndices = list(kernelResultIterator.multi_index)
                bestResolution = resolutionForKernelResult

        self.model.multivalueValueIndices = bestMultivalueIndices

    @pyqtSlot()
    def _uiClickImportResults(self) -> None:
        """ Imports previously saved simulation results from a user-picked file. """

        path, _ = QFileDialog.getOpenFileName(
            self.widget,
            caption="Open results file",
            filter="All compatible files (*.json;*.bin);;JSON files (*.json);;Binary files (*.bin)",
            directory=UserFileDirs.SavedResults
        )

        if path:  # Check whether a file was picked
            try:
                if path.endswith(".bin"):
                    # Open binary pickle file
                    confirmation_result = QMessageBox.warning(
                        self.widget, "Security Warning",
                        "You're about to open a simulation stored in binary format. Since" +
                        " data from this type of file can execute arbitrary code and thus is" +
                        " a security risk, it is highly recommended that you only proceed if" +
                        " you created the file yourself, or if it comes from a source that"
                        " you trust." +
                        "\n\nContinue loading the file?",
                        QMessageBox.Yes | QMessageBox.No, defaultButton=QMessageBox.No)

                    if confirmation_result != QMessageBox.Yes:
                        return

                    with open(path, "rb") as pickleFile:
                        persistentContainer = rebuild_dataclass(pickle.load(pickleFile))
                else:
                    # Open JSON file
                    with open(path, "r") as jsonFile:
                        json = jsonFile.read()

                    persistentContainer = PersistentContainer[
                        SimulationResults
                    ].from_json_with_converted_dicts(
                        json, SimulationResults
                    )

                for warning in persistentContainer.validate():
                    QMessageBox.warning(self.widget, "Results load warning", warning)

                self.model = persistentContainer.data
            except Exception as e:
                print(format_exc())
                QMessageBox.critical(self.widget, "Results load error", str(e))

    @pyqtSlot()
    def _uiClickExportResults(self) -> None:
        """ Exports the current simulation results to a user-picked file. """

        path, _ = QFileDialog.getSaveFileName(
            self.widget,
            caption="Save results file",
            filter="JSON files (*.json);;Binary files (*.bin)",
            directory=UserFileDirs.SavedResults
        )

        if path:  # Check whether a file was picked
            if path.endswith(".bin"):
                # Cache all simulations and save binary pickle file
                sampleImage = self.model.sampleImage
                self.model.results.cache_all(
                    sampleImage.id if sampleImage is not None else None,
                    sampleImage.imageArr if sampleImage is not None else None
                )

                persistentContainer = PersistentContainer(
                    clear_signals(deepcopy(self.model.results))
                )
                with open(path, "wb") as pickleFile:
                    pickle.dump(persistentContainer, pickleFile, pickle.HIGHEST_PROTOCOL)
            else:
                # Save JSON file
                persistentContainer = PersistentContainer(self.model.results)

                with open(path, "w") as jsonFile:
                    jsonFile.write(persistentContainer.to_json())
