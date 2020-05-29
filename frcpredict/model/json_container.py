from dataclasses import dataclass
from dataclasses_json import dataclass_json
from packaging import version
from typing import Any, Type

import frcpredict


@dataclass_json
@dataclass
class JsonContainer:
    """
    This dataclass is intended to be a wrapper for objects imported/exported in JSON format. It
    contains the wrapped object itself, a name, the type of the object (for ensuring that the user
    is importing the right thing), and the name as well as version of the program the file was
    exported with (for checking compatibility).
    """

    name: str
    data: Any
    data_type: str
    program_name: str
    program_version: str

    def __init__(self, name: str, data: Any) -> None:
        self.name = name
        self.data = data
        self.data_type = type(data).__name__
        self.program_name = frcpredict.__title__
        self.program_version = frcpredict.__version__

    def validate(self, expected_type: Type) -> bool:
        """
        Returns true if and only if the data type in the file matches the expected type, the file
        was generated with the same program as the one it's being run on, and the version of the
        program that the file was generated with is not higher than the current program version.
        """

        return (self.data_type == expected_type.__name__ and
                self.program_name == frcpredict.__title__ and
                version.parse(self.program_version) <= version.parse(frcpredict.__version__))
