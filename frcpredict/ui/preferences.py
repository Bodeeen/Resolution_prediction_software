import os
from dataclasses import dataclass

from dataclasses_json import dataclass_json
from PySignal import Signal

from frcpredict.model import PersistentContainer
from frcpredict.util import dataclass_internal_attrs, dataclass_with_properties, observable_property
from .util.file_utils import subOfUserFilesDir


@dataclass_json
@dataclass_with_properties
@dataclass_internal_attrs(basicFieldChanged=Signal)
@dataclass
class Preferences:
    """
    A description of the persistent preferences that the user should be able to change in the
    program.
    """

    precacheFrcCurves: bool = observable_property(
        "_precacheFrcCurves", default=True, signal_name="basicFieldChanged"
    )

    precacheExpectedImages: bool = observable_property(
        "_precacheExpectedImages", default=False, signal_name="basicFieldChanged"
    )

    cacheKernels2D: bool = observable_property(
        "_cacheKernels2D", default=True, signal_name="basicFieldChanged"
    )

    # Methods
    @staticmethod
    def initFile() -> None:
        """ Creates the preferences file if it does not exist. """
        if not os.path.isfile(_preferencesFilePath):
            Preferences.save(Preferences())
        else:
            Preferences.get()

    @staticmethod
    def get() -> "Preferences":
        """
        Returns the persistent preferences object. It will be loaded from the disk if that has not
        already been done.
        """

        global _preferences

        if _preferences is None:
            with open(_preferencesFilePath, "r") as jsonFile:
                json = jsonFile.read()
                persistentContainer = PersistentContainer[
                    Preferences
                ].from_json_with_converted_dicts(
                    json, Preferences
                )
                _preferences = persistentContainer.data

        return _preferences

    @staticmethod
    def save(preferences: "Preferences"):
        """ Saves the given persistent preferences object to memory as well as the disk. """

        global _preferences
        _preferences = preferences

        with open(_preferencesFilePath, "w") as jsonFile:
            persistentContainer = PersistentContainer[Preferences](preferences)
            jsonFile.write(persistentContainer.to_json(indent=2))


_preferencesFilePath = subOfUserFilesDir("preferences.json")
_preferences = None
