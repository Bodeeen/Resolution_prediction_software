from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config as json_config
from marshmallow import fields
import numpy as np
from osgeo import gdal_array
from skimage.transform import resize
from typing import Optional, Union, List, Dict, Type

from frcpredict.util import (
    get_canvas_params,
    gaussian_test1, doughnut_test1, airy_test1,
    digital_pinhole_test1
)


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
    def from_npy_file(path: str) -> None:
        """
        Loads a 2D array from an .npy file. The file is expected to contain a float array that is
        of the shape (width, height) and has values within the range [-1, 1].
        """
        return Array2DPatternData(value=np.load(path))

    @staticmethod
    def from_image_file(path: str) -> None:
        """ Loads a 2D array from an image file. """
        raster_array = gdal_array.LoadFile(path)  # Load image as numpy array
        
        if np.issubdtype(raster_array.dtype, np.integer):
            raster_array = raster_array / 255.0  # Image has int values; normalise

        if raster_array.ndim > 2:
            # Image has multiple channels; average the values over the channels
            raster_array = np.mean(raster_array, axis=0)

        return Array2DPatternData(value=raster_array)


@dataclass_json
@dataclass
class GaussianPatternData(PatternData):
    amplitude: float = 1
    fwhm: float = 480

    # Methods
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        return gaussian_test1(amplitude=self.amplitude, fwhm=self.fwhm, pixels_per_nm=pixels_per_nm)

    def __str__(self) -> str:
        return f"Gaussian; amplitude = {self.amplitude}, FWHM = {self.fwhm} nm"


@dataclass_json
@dataclass
class DoughnutPatternData(PatternData):
    periodicity: float = 540

    # Methods
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        return doughnut_test1(periodicity=self.periodicity, pixels_per_nm=pixels_per_nm)

    def __str__(self) -> str:
        return f"Doughnut; periodicity = {self.periodicity} nm"


@dataclass_json
@dataclass
class AiryPatternData(PatternData):
    amplitude: float = 1
    fwhm: float = 240

    # Methods
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        return airy_test1(amplitude=self.amplitude, fwhm=self.fwhm, pixels_per_nm=pixels_per_nm)

    def __str__(self) -> str:
        return f"Airy; amplitude = {self.amplitude}, FWHM = {self.fwhm} nm"


@dataclass_json
@dataclass
class DigitalPinholePatternData(PatternData):
    fwhm: float = 240

    # Methods
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        return digital_pinhole_test1(fwhm=self.fwhm, pixels_per_nm=pixels_per_nm)

    def __str__(self) -> str:
        return f"Digital pinhole; FWHM = {self.fwhm} nm"
