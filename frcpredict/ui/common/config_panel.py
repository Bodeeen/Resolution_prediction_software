import os
from copy import deepcopy
from traceback import format_exc
from typing import Any, Optional, Callable

from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QMenu

from frcpredict.model import PersistentContainer
from frcpredict.ui import BaseWidget


class ConfigPanelWidget(BaseWidget):
    """
    A widget where the user may load and save configurations.

    setModelType, setValueGetter, and setValueSetter must be called to set values required for this
    widget to function.
    """

    # Signals
    dataLoaded = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        self._modelType = None
        self._startDirectory = None
        self._presetsDirectory = None
        self._valueGetter = None
        self._valueSetter = None
        self._modified = False

        super().__init__(__file__, *args, **kwargs)
        self.setLoadedPath(None)
        self.setFieldName("Config")

        # Prepare UI elements
        self.scrLoadedConfigName.setMaximumHeight(self.lblLoadedConfigName.height())
        self.btnPresets.setVisible(False)
        self.sepPresets.setVisible(False)

        self.setFocusPolicy(Qt.TabFocus)
        self.setFocusProxy(self.btnLoadFile)

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
        self._modified = False
        self.btnSaveFile.setEnabled(loadedPath is not None)
        self.lblLoadedConfigName.setText(
            os.path.basename(loadedPath) if loadedPath else "(no file loaded)"
        )

    def modelType(self) -> type:
        """ Returns the type of model that is intended to be loaded/saved. """
        return self._modelType

    def setModelType(self, modelType: type) -> None:
        """ Sets the type of model that is intended to be loaded/saved. """
        self._modelType = modelType

    def isModified(self) -> bool:
        """ Returns whether the current configuration has been modified since it was loaded. """
        return self._modified

    def setModifiedFlag(self) -> None:
        """ Sets that the current configuration has been modified since it was loaded. """

        if self._modified:
            return

        self.lblLoadedConfigName.setText(f"*{self.lblLoadedConfigName.text()}")
        self._modified = True

    def clearModifiedFlag(self) -> None:
        """ Unsets that the current configuration has been modified since it was loaded. """

        if not self._modified:
            return

        self.setLoadedPath(self.loadedPath())
        self._modified = False

    def presetsDirectory(self) -> Optional[str]:
        """
        Returns the directory that contains the presets that are available to load from the panel.
        """
        return self._presetsDirectory

    def setPresetsDirectory(self, presetsDirectory: Optional[str]) -> None:
        """
        Sets a directory that contains presets that will be available to load from the panel.
        """

        if self._modelType is None:
            raise Exception("Model type not set for config panel.")

        self._presetsDirectory = presetsDirectory
        hasPresets = False

        if presetsDirectory and os.path.isdir(presetsDirectory):
            actionMenu = QMenu()

            for presetFileName in os.listdir(presetsDirectory):
                presetFilePath = os.path.join(presetsDirectory, presetFileName)
                try:
                    with open(presetFilePath, "r") as presetFile:
                        json = presetFile.read()

                        preset = PersistentContainer[
                            self.modelType()
                        ].from_json_with_converted_dicts(
                            json, self.modelType()
                        )

                        if not preset.name:
                            preset.name = presetFileName  # If no name was defined, use file name

                        actionMenu.addAction(preset.name, lambda: self._loadPreset(preset))
                        hasPresets = True
                except Exception as e:
                    print(format_exc())
                    QMessageBox.warning(
                        self, "Preset initialization error",
                        f"Failed to init preset from file {presetFilePath}: {e}"
                    )

            self.btnPresets.setMenu(actionMenu if hasPresets else None)
        else:
            self.btnPresets.setMenu(None)

        self.btnPresets.setVisible(hasPresets)
        self.sepPresets.setVisible(hasPresets)

    def startDirectory(self) -> Optional[str]:
        """
        Returns the directory that the config panel will first show when letting the user pick a
        file to load or save.
        """
        return self._startDirectory

    def setStartDirectory(self, startDirectory: Optional[str]) -> None:
        """
        Sets the directory that the config panel will first show when letting the user pick a file
        to load or save.
        """
        self._startDirectory = startDirectory

    def setValueGetter(self, valueGetter: Callable[[], Any]) -> None:
        """
        Sets the function that the config panel will call to read the model when saving the
        current configuration. The function must have no required arguments, and it must return the
        model in its entirety.
        """
        self._valueGetter = valueGetter

    def setValueSetter(self, valueSetter: Callable[[Any], None]) -> None:
        """
        Sets the function that the config panel will call to update the model when a configuration
        file is loaded. The function must have one argument, which the config panel will send the
        new model through.
        """
        self._valueSetter = valueSetter

    # Internal methods
    def _loadUserConfigFromFile(self, path: str):
        """ Loads a JSON file as the current configuration. """

        if self._modelType is None or self._valueSetter is None:
            raise Exception("Model type or value setter not set for config panel.")

        with open(path, "r") as jsonFile:
            try:
                json = jsonFile.read()

                persistentContainer = PersistentContainer[
                    self.modelType()
                ].from_json_with_converted_dicts(
                    json, self.modelType()
                )

                self._loadPersistentContainer(persistentContainer, path)
            except Exception as e:
                print(format_exc())
                QMessageBox.critical(self, "Configuration load error", str(e))

    def _loadPreset(self, preset: PersistentContainer) -> None:
        """ Loads a preset as the current configuration. """
        self._loadPersistentContainer(deepcopy(preset))
        self.lblLoadedConfigName.setText(
            preset.name + " (preset)" if preset.name else "(preset)"
        )

    def _loadPersistentContainer(self, persistentContainer: PersistentContainer,
                                 loadedPath: Optional[str] = None) -> None:
        """ Loads a persistent container as the current configuration. """

        for warning in persistentContainer.validate():
            QMessageBox.warning(self, "Configuration load warning", warning)

        self._valueSetter(persistentContainer.data)
        self.setLoadedPath(loadedPath)
        self.dataLoaded.emit()

    def _saveUserConfigToFile(self, path: str):
        """ Saves the current configuration to a JSON file. """

        if self._modelType is None or self._valueGetter is None:
            raise Exception("Model type or value getter not set for config panel.")

        with open(path, "w") as jsonFile:
            try:
                persistentContainer = PersistentContainer[self.modelType()](self._valueGetter())
                jsonFile.write(persistentContainer.to_json(indent=2))

                self.setLoadedPath(path)
            except Exception as e:
                QMessageBox.critical(self, "Configuration save error", str(e))

    # Event handling
    @pyqtSlot()
    def _onClickLoadFile(self) -> None:
        """ Loads a configuration file chosen by the user as the current configuration. """

        path, _ = QFileDialog.getOpenFileName(
            self,
            caption="Open Configuration File",
            filter="JSON files (*.json)",
            directory=(os.path.dirname(self.loadedPath()) if self.loadedPath() is not None
                       else self.startDirectory())
        )

        if path:  # Check whether a file was picked
            self._loadUserConfigFromFile(path)

    @pyqtSlot()
    def _onClickSaveFile(self) -> None:
        """ Saves the current configuration to the currently loaded configuration file. """

        path = self.loadedPath()
        if path:
            self._saveUserConfigToFile(path)
        else:
            self._onClickSaveFileAs()

    @pyqtSlot()
    def _onClickSaveFileAs(self) -> None:
        """ Saves the current configuration to a configuration file chosen by the user. """

        path, _ = QFileDialog.getSaveFileName(
            self,
            caption="Save Configuration File",
            filter="JSON files (*.json)",
            directory=self.loadedPath() if self.loadedPath() is not None else self.startDirectory()
        )

        if path:  # Check whether a file was picked
            self._saveUserConfigToFile(path)
