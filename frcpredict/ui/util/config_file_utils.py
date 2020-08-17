import os
from abc import ABC

_basePresetFilesDir = os.path.join("data", "presets")
_baseUserFilesDir = "user_files"


def subdirOfPresetFilesDir(subdir: str) -> str:
    return os.path.join(os.getcwd(), _basePresetFilesDir, subdir)


def subdirOfUserFilesDir(subdir: str) -> str:
    return os.path.join(os.getcwd(), _baseUserFilesDir, subdir)


def initUserFilesIfNeeded() -> None:
    for userFileDir in UserFileDirs.list():
        os.makedirs(userFileDir, exist_ok=True)


class ConfigFileDirs(ABC):
    @classmethod
    def list(cls):
        return [cls.__dict__.get(name) for name in dir(cls) if (
            not callable(getattr(cls, name)) and not name.startswith("_")
        )]


class PresetFileDirs(ConfigFileDirs):
    FluorophoreSettings: str = subdirOfPresetFilesDir("fluorophore_settings")
    ImagingSystemSettings: str = subdirOfPresetFilesDir("imaging_system_settings")
    PulseScheme: str = subdirOfPresetFilesDir("pulse_scheme")
    SampleProperties: str = subdirOfPresetFilesDir("sample_properties")
    CameraProperties: str = subdirOfPresetFilesDir("camera_properties")
    RunInstance: str = subdirOfPresetFilesDir("run_instance")


class UserFileDirs(ConfigFileDirs):
    FluorophoreSettings: str = subdirOfUserFilesDir("fluorophore_settings")
    ImagingSystemSettings: str = subdirOfUserFilesDir("imaging_system_settings")
    PulseScheme: str = subdirOfUserFilesDir("pulse_scheme")
    SampleProperties: str = subdirOfUserFilesDir("sample_properties")
    CameraProperties: str = subdirOfUserFilesDir("camera_properties")
    RunInstance: str = subdirOfUserFilesDir("run_instance")
    SavedResults: str = subdirOfUserFilesDir("saved_results")
    SimulatedImages: str = subdirOfUserFilesDir("simulated_images")
