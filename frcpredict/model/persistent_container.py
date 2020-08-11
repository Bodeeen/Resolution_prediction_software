from dataclasses import dataclass
from typing import Optional, TypeVar, Generic, List

from dataclasses_json import dataclass_json
from packaging import version

import frcpredict
from frcpredict.util import is_dataclass_instance, recursive_field_iter

DataType = TypeVar("DataType")


@dataclass_json
@dataclass
class PersistentContainer(Generic[DataType]):
    """
    This dataclass is intended to be a wrapper for objects imported/exported in primarily (but not
    exclusively) JSON format. It contains the wrapped object itself, a name, the type of the object
    (for ensuring that the user is importing the right thing), and the name as well as version of
    the program the file was exported with (for checking compatibility).
    """

    data: DataType
    data_type: str
    program_name: str
    program_version: str

    def __init__(self, data: DataType, data_type: Optional[type] = None,
                 program_name: Optional[str] = None, program_version: Optional[str] = None) -> None:
        self.data = data
        self.data_type = type(data).__name__ if data_type is None else data_type
        self.program_name = frcpredict.__title__ if program_name is None else program_name
        self.program_version = frcpredict.__version__ if program_version is None else program_version

    def validate(self) -> List[str]:
        """ Validates the loaded JSON file contents and returns a list of any issues found. """

        warnings = []

        if self.program_name != frcpredict.__title__:
            warnings.append(
                f"Data was created with program \"{self.program_name}\" which is different from what you are using (\"{frcpredict.__title__}\")")

        if version.parse(self.program_version) > version.parse(frcpredict.__version__):
            warnings.append(
                f"Data was created with program version \"{self.program_version}\" which is newer than what you are using (\"{frcpredict.__version__}\")")

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

        persistent_container = PersistentContainer[data_type].from_json(json, infer_missing=True)

        if persistent_container.data is None:
            raise Exception("Incompatible JSON: Did not contain data object.")

        if isinstance(persistent_container.data, dict):
            try:
                persistent_container.data = data_type.from_dict(persistent_container.data)
            except Exception as e:
                if persistent_container.data_type != data_type.__name__:
                    raise Exception(
                        f"Incompatible JSON: Described data type \"{persistent_container.data_type}\" did not match expected data type \"{data_type.__name__}\"")
                else:
                    raise e

        # Upgrade the data structure if this JSON was created by an previous version of the program.
        # The data structure, which can be seen as a tree, is upgraded in a bottom-up manner.
        parsed_json_version = version.parse(persistent_container.program_version)
        if (persistent_container.program_name != frcpredict.__title__
                and parsed_json_version < version.parse(frcpredict.__version__)):
            for field_value in reversed(recursive_field_iter(persistent_container)):
                if is_dataclass_instance(field_value):
                    if hasattr(field_value, "upgrade") and callable(field_value.upgrade):
                        field_value.upgrade(parsed_json_version)

        # Return
        return persistent_container
