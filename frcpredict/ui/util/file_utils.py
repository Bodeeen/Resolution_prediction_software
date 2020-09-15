import os
from abc import ABC


def getUserDir() -> str:
    """
    Returns the user's documents folder if they are using a Windows system, or their home folder if
    they are using another operating system.
    """

    if os.name == "nt":  # Windows system, try to return documents directory
        try:
            import ctypes.wintypes
            CSIDL_PERSONAL = 5  # Documents
            SHGFP_TYPE_CURRENT = 0  # Current value

            buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
            ctypes.windll.shell32.SHGetFolderPathW(0, CSIDL_PERSONAL, 0, SHGFP_TYPE_CURRENT, buf)

            return buf.value
        except ImportError:
            pass

    return os.path.expanduser("~")  # Non-Windows system, return home directory


_basePresetFilesDir = os.path.join("data", "presets")
_baseUserFilesDir = os.path.join(getUserDir(), "frcpredict")
_baseUserConfigFilesDir = os.path.join(_baseUserFilesDir, "saved_configs")


def subOfPresetFilesDir(subdir: str) -> str:
    """ Returns the sub-file or sub-directory with the given name, in the presets directory. """
    return os.path.join(os.getcwd(), _basePresetFilesDir, subdir)


def subOfUserFilesDir(subdir: str) -> str:
    """ Returns the sub-file or sub-directory with the given name, in the user files directory. """
    return os.path.join(os.getcwd(), _baseUserFilesDir, subdir)


def subOfUserConfigFilesDir(subdir: str) -> str:
    """
    Returns the sub-file or sub-directory with the given name, in the user's saved configuration
    files directory.
    """
    return os.path.join(os.getcwd(), _baseUserConfigFilesDir, subdir)


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
    DetectorProperties: str = subOfPresetFilesDir("detector_properties")
    RunInstance: str = subOfPresetFilesDir("run_instance")


class UserFileDirs(ConfigFileDirs):
    """
    Catalog of directories that contain user-saved configuration files.
    """

    FluorophoreSettings: str = subOfUserConfigFilesDir("fluorophore_settings")
    ImagingSystemSettings: str = subOfUserConfigFilesDir("imaging_system_settings")
    PulseScheme: str = subOfUserConfigFilesDir("pulse_scheme")
    SampleProperties: str = subOfUserConfigFilesDir("sample_properties")
    DetectorProperties: str = subOfUserConfigFilesDir("detector_properties")
    RunInstance: str = subOfUserConfigFilesDir("run_instance")

    SavedResults: str = subOfUserFilesDir("saved_results")
    SimulatedData: str = subOfUserFilesDir("simulated_data")
