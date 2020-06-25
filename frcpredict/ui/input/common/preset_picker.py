from dataclasses import fields
from typing import Any, Optional, Callable
import os

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMessageBox, QFileDialog

from frcpredict.model import JsonContainer
from frcpredict.ui import BaseWidget
import frcpredict.ui.resources


class PresetPickerWidget(BaseWidget):
    """
    A widget where the user may load and save configuration presets.

    setModelType, setValueGetter, and setValueSetter must be called to set values required for this
    widget to function.
    """

    # Signals
    dataLoaded = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        self._modelType = None
        self._startDirectory = None
        self._valueGetter = None
        self._valueSetter = None

        super().__init__(__file__, *args, **kwargs)
        self.setLoadedPath(None)
        self.setFieldName("Config")

        # Connect signals
        self.btnLoadFile.clicked.connect(self._onClickLoadFile)
        self.btnSaveFile.clicked.connect(self._onClickSaveFile)
        self.btnSaveFileAs.clicked.connect(self._onClickSaveFileAs)

    def fieldName(self) -> str:
        """ Returns the displayed name of the field, e.g. "Global config". """
        return self._fieldName

    def setFieldName(self, fieldName: str) -> None:
        """ Sets the displayed name of the field, e.g. "Global config". """
        self._fieldName = fieldName
        self.lblFieldName.setText(f"{fieldName}:")

    def loadedPath(self) -> Optional[str]:
        """ Returns the displayed path to the currently loaded file. """
        return self._loadedPath

    def setLoadedPath(self, loadedPath: Optional[str]) -> None:
        """ Sets the displayed path to the currently loaded file. """
        self._loadedPath = loadedPath
        self.btnSaveFile.setEnabled(loadedPath is not None)
        self.lblLoadedConfigName.setText(os.path.basename(loadedPath) if loadedPath else "(no file loaded)")

    def modelType(self) -> type:
        """ Returns the type of model that is intended to be loaded/saved. """
        return self._modelType

    def setModelType(self, modelType: type) -> None:
        """ Sets the type of model that is intended to be loaded/saved. """
        self._modelType = modelType

    # TODO: Use this function for indicating when the current configuration has unsaved changes
    #def setModifiedFlag(self, modified: bool) -> None:
        #self.lblConfigName.setText("*(custom)" if modified else "(custom)")
        #newFont = QFont(self.lblConfigName.font())
        # newFont.setItalic(modified)
        # self.lblConfigName.setFont(newFont)

    def startDirectory(self) -> str:
        """
        Returns the directory that the preset picker will first show when letting the user pick a
        file to load or save.
        """
        return self._startDirectory

    def setStartDirectory(self, startDirectory: str) -> None:
        """
        Sets the directory that the preset picker will first show when letting the user pick a file
        to load or save.
        """
        self._startDirectory = startDirectory

    def setValueGetter(self, valueGetter: Callable[[], Any]) -> None:
        """
        Sets the function that the preset picker will call to read the model when saving the
        current configuration. The function must have no required arguments, and it must return the
        model in its entirety.
        """
        self._valueGetter = valueGetter

    def setValueSetter(self, valueSetter: Callable[[Any], None]) -> None:
        """
        Sets the function that the preset picker will call to update the model when a configuration
        file is loaded. The function must have one argument, which the preset picker will send the
        new model through.
        """
        self._valueSetter = valueSetter

    # Internal methods
    def _loadFromFile(self, path: str):
        """
        Loads a JSON file as the current configuration.
        TODO: Might be better to have this logic in model.
        """

        if self._modelType is None or self._valueSetter is None:
            raise Exception("Model type or value setter not set for preset picker.")

        with open(path, "r") as jsonFile:
            try:
                json = jsonFile.read()

                jsonContainer = JsonContainer[self.modelType()].from_json_with_converted_dicts(
                    json, self.modelType()
                )

                for warning in jsonContainer.validate():
                    QMessageBox.warning(self, "Configuration load warning", warning)

                self._valueSetter(jsonContainer.data)

                self.setLoadedPath(path)
                self.dataLoaded.emit()
            except Exception as e:
                QMessageBox.critical(self, "Configuration load error", str(e))

    def _saveToFile(self, path: str):
        """
        Saves the current configuration to a JSON file.
        TODO: Might be better to have this logic in model.
        """

        if self._modelType is None or self._valueGetter is None:
            raise Exception("Model type or value getter not set for preset picker.")
        
        with open(path, "w") as jsonFile:
            try:
                jsonContainer = JsonContainer[self.modelType()](self._valueGetter())
                jsonFile.write(jsonContainer.to_json(indent=2))

                self.setLoadedPath(path)
            except Exception as e:
                QMessageBox.critical(self, "Configuration save error", str(e))

    # Event handling
    @pyqtSlot()
    def _onClickLoadFile(self) -> None:
        """ Loads a configuration file chosen by the user as the current configuration. """

        path, _ = QFileDialog.getOpenFileName(
            self,
            caption="Open configuration file",
            filter="JSON files (*.json)",
            directory=self.startDirectory()
        )

        if path:  # Check whether a file was picked
            self._loadFromFile(path)

    @pyqtSlot()
    def _onClickSaveFile(self) -> None:
        """ Saves the current configuration to the currently loaded configuration file. """

        path = self.loadedPath()
        if path:
            self._saveToFile(path)
        else:
            self._onClickSaveFileAs()

    @pyqtSlot()
    def _onClickSaveFileAs(self) -> None:
        """ Saves the current configuration to a configuration file chosen by the user. """

        path, _ = QFileDialog.getSaveFileName(
            self,
            caption="Save configuration file",
            filter="JSON files (*.json)",
            directory=self.startDirectory()
        )

        if path:  # Check whether a file was picked
            self._saveToFile(path)
