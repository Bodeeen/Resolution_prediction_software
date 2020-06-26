from dataclasses import dataclass
from dataclasses_json import dataclass_json
from packaging import version
from typing import TypeVar, Generic, List

import frcpredict


DataType = TypeVar('DataType')


@dataclass_json
@dataclass
class JsonContainer(Generic[DataType]):
    """
    This dataclass is intended to be a wrapper for objects imported/exported in JSON format. It
    contains the wrapped object itself, a name, the type of the object (for ensuring that the user
    is importing the right thing), and the name as well as version of the program the file was
    exported with (for checking compatibility).
    """

    data: DataType
    data_type: str
    program_name: str
    program_version: str

    def __init__(self, data: DataType, data_type: type = None,
                 program_name: str = None, program_version: str = None) -> None:
        self.data = data
        self.data_type = type(data).__name__ if data_type is None else data_type
        self.program_name = frcpredict.__title__ if program_name is None else program_name
        self.program_version = frcpredict.__version__ if program_version is None else program_version

    def validate(self) -> List[str]:
        """ Validates the loaded JSON file contents and returns a list of any issues found. """

        warnings = []

        if self.program_name != frcpredict.__title__:
            warnings.append(
                f"Data was created with program \"{self.program_name}\" which is different yours \"{frcpredict.__title__}\"")

        if version.parse(self.program_version) > version.parse(frcpredict.__version__):
            warnings.append(
                f"Data was created with program version \"{self.program_version}\" which is newer than yours \"{frcpredict.__version__}\"")

        if self.data_type != type(self.data).__name__ and not isinstance(self.data, dict):
            warnings.append(
                f"Actual data type \"{type(self.data).__name__}\" did not match described data type \"{self.data_type}\"")

        return warnings

    @staticmethod
    def from_json_with_converted_dicts(json: str, data_type: type):
        """
        Returns a model from JSON. Built upon the from_json function of the dataclasses-json
        library, this method will properly handle the potential different types of data in the data
        field; instead of returning it as a dict, it will be returned as an instance of its
        corresponding dataclass.
        """

        json_container = JsonContainer[data_type].from_json(json)

        if isinstance(json_container.data, dict):
            try:
                json_container.data = data_type.from_dict(json_container.data)
            except Exception as e:
                if json_container.data_type != data_type.__name__:
                    raise Exception(
                        f"Described data type \"{json_container.data_type}\" did not match expected data type \"{data_type.__name__}\"")
                else:
                    raise e

        return json_container
