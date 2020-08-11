import os
from typing import Tuple

_data_dir = os.path.join(os.getcwd(), "data")
_samples_dir = os.path.join(_data_dir, "sample_structures")


def get_sample_structure_data_dir_names():
    """ Returns the names of all sample structure directories. """
    return os.listdir(_samples_dir)


def get_sample_structure_data_file_paths(sample_dir_name: str) -> Tuple[str, str]:
    """
    Returns the properties file path and image file path respectively of the sample structure with
    with the given directory name.
    """

    sample_dir_path = os.path.join(_samples_dir, sample_dir_name)
    properties_file_path = os.path.join(sample_dir_path, "properties.json")
    image_file_path = os.path.join(sample_dir_path, "image.tiff")

    return properties_file_path, image_file_path
