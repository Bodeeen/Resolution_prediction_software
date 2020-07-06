from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union

import numpy as np
from dataclasses_json import dataclass_json, config as json_config
from marshmallow import fields
from skimage.io import imread
from skimage.transform import resize

from frcpredict.util import (
    get_canvas_params, rangeable_field,
    gaussian_test1, doughnut_test1, airy_test1,
    digital_pinhole_test1, physical_pinhole_test1
)
from .value_range import ValueRange


@dataclass_json
@dataclass
class PatternData(ABC):
    @abstractmethod
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        """ Returns a numpy array representation of the pattern data. """
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass_json
@dataclass
class Array2DPatternData(PatternData):
    value: np.ndarray = field(
        default=np.zeros((1, 1)),
        metadata=json_config(
            encoder=lambda x: np.array(x).astype(float).tolist(),
            decoder=np.array,
            mm_field=fields.List
        ))

    # Methods
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        _, canvas_side_length_px = get_canvas_params(pixels_per_nm)
        canvas_size = (canvas_side_length_px, canvas_side_length_px)

        if self.value.shape != canvas_size:
            # TODO: This assumes that the loaded pattern has an inner radius of the same length as
            #       the constant _canvas_inner_radius_nm; this may not always be correct
            return resize(self.value, canvas_size, order=3)
        else:
            return self.value

    def __str__(self) -> str:
        if np.any(self.value):
            return "Loaded from file"
        else:  # All zeros in value
            return "Empty pattern"

    @staticmethod
    def from_npy_file(path: str) -> PatternData:
        """
        Loads a 2D array from an .npy file. The file is expected to contain a float array that is
        of the shape (width, height).
        """
        return Array2DPatternData(value=np.load(path))

    @staticmethod
    def from_image_file(path: str) -> PatternData:
        """ Loads a 2D array from an image file. """
        raster_array = imread(path, as_gray=True)  # Load image as numpy array

        if np.issubdtype(raster_array.dtype, np.integer):
            raster_array = raster_array / 255.0  # Image has int values; normalise

        return Array2DPatternData(value=raster_array)


@dataclass_json
@dataclass
class GaussianPatternData(PatternData):
    amplitude: Union[float, ValueRange[float]] = rangeable_field(default=1.0)
    fwhm: Union[float, ValueRange[float]] = rangeable_field(default=480.0)  # nanometres

    # Methods
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        return gaussian_test1(amplitude=self.amplitude, fwhm=self.fwhm, pixels_per_nm=pixels_per_nm)

    def __str__(self) -> str:
        return f"Gaussian; amplitude = {self.amplitude}, FWHM = {self.fwhm} nm"


@dataclass_json
@dataclass
class DoughnutPatternData(PatternData):
    periodicity: Union[float, ValueRange[float]] = rangeable_field(default=540.0)  # nanometres

    # Methods
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        return doughnut_test1(periodicity=self.periodicity, pixels_per_nm=pixels_per_nm)

    def __str__(self) -> str:
        return f"Doughnut; periodicity = {self.periodicity} nm"


@dataclass_json
@dataclass
class AiryPatternData(PatternData):
    amplitude: Union[float, ValueRange[float]] = rangeable_field(default=1.0)
    fwhm: Union[float, ValueRange[float]] = rangeable_field(default=240.0)  # nanometres

    # Methods
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        return airy_test1(amplitude=self.amplitude, fwhm=self.fwhm, pixels_per_nm=pixels_per_nm)

    def __str__(self) -> str:
        return f"Airy; amplitude = {self.amplitude}, FWHM = {self.fwhm} nm"


@dataclass_json
@dataclass
class DigitalPinholePatternData(PatternData):
    fwhm: Union[float, ValueRange[float]] = 240.0  # nanometres

    # Methods
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        return digital_pinhole_test1(fwhm=self.fwhm, pixels_per_nm=pixels_per_nm)

    def __str__(self) -> str:
        return f"Digital pinhole; FWHM = {self.fwhm} nm"


@dataclass_json
@dataclass
class PhysicalPinholePatternData(PatternData):
    radius: Union[float, ValueRange[float]] = rangeable_field(default=100.0)  # nanometres

    # Methods
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        return physical_pinhole_test1(radius=self.radius, pixels_per_nm=pixels_per_nm)

    def __str__(self) -> str:
        return f"Physical pinhole; radius = {self.radius} nm"
