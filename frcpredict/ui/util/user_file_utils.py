import os

_baseUserFilesDir = "user_files"


def subdirOfUserFilesDir(subdir: str) -> str:
    return os.path.join(os.getcwd(), _baseUserFilesDir, subdir)


def initUserFilesIfNeeded() -> None:
    for userFileDir in UserFileDirs.list():
        os.makedirs(userFileDir, exist_ok=True)


class UserFileDirs:
    FluorophoreSettings: str = subdirOfUserFilesDir("fluorophore_settings")
    ImagingSystemSettings: str = subdirOfUserFilesDir("imaging_system_settings")
    PulseScheme: str = subdirOfUserFilesDir("pulse_scheme")
    SampleProperties: str = subdirOfUserFilesDir("sample_properties")
    CameraProperties: str = subdirOfUserFilesDir("camera_properties")

    RunInstance: str = subdirOfUserFilesDir("run_instance")
    SavedResults: str = subdirOfUserFilesDir("saved_results")

    @staticmethod
    def list():
        return [UserFileDirs.__dict__.get(name) for name in dir(UserFileDirs) if (
            not callable(getattr(UserFileDirs, name)) and not name.startswith("_")
        )]
