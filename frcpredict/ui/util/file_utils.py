import os
from abc import ABC

_basePresetFilesDir = os.path.join("data", "presets")
_baseUserFilesDir = "user_files"


def subOfPresetFilesDir(subdir: str) -> str:
    """ Returns the sub-file or sub-directory with the given name, in the presets directory. """
    return os.path.join(os.getcwd(), _basePresetFilesDir, subdir)


def subOfUserFilesDir(subdir: str) -> str:
    """ Returns the sub-file or sub-directory with the given name, in the user files directory. """
    return os.path.join(os.getcwd(), _baseUserFilesDir, subdir)


def initUserFilesIfNeeded() -> None:
    """ Initializes all directories and files that will be used to store the user's data. """

    # Create directories if they don't exist
    for userFileDir in UserFileDirs.list():
        os.makedirs(userFileDir, exist_ok=True)

    # Init preferences file
    from frcpredict.ui import Preferences
    Preferences.initFile()


class ConfigFileDirs(ABC):
    """
    Base class for directory catalog classes.
    """

    @classmethod
    def list(cls):
        """ Returns all directories in the catalog. """
        return [cls.__dict__.get(name) for name in dir(cls) if (
            not callable(getattr(cls, name)) and not name.startswith("_")
        )]


class PresetFileDirs(ConfigFileDirs):
    """
    Catalog of directories that contain preset configuration files.
    """

    FluorophoreSettings: str = subOfPresetFilesDir("fluorophore_settings")
    ImagingSystemSettings: str = subOfPresetFilesDir("imaging_system_settings")
    PulseScheme: str = subOfPresetFilesDir("pulse_scheme")
    SampleProperties: str = subOfPresetFilesDir("sample_properties")
    CameraProperties: str = subOfPresetFilesDir("camera_properties")
    RunInstance: str = subOfPresetFilesDir("run_instance")


class UserFileDirs(ConfigFileDirs):
    """
    Catalog of directories that contain user-saved configuration files.
    """

    FluorophoreSettings: str = subOfUserFilesDir("fluorophore_settings")
    ImagingSystemSettings: str = subOfUserFilesDir("imaging_system_settings")
    PulseScheme: str = subOfUserFilesDir("pulse_scheme")
    SampleProperties: str = subOfUserFilesDir("sample_properties")
    CameraProperties: str = subOfUserFilesDir("camera_properties")
    RunInstance: str = subOfUserFilesDir("run_instance")
    SavedResults: str = subOfUserFilesDir("saved_results")
    SimulatedData: str = subOfUserFilesDir("simulated_data")
