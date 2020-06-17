from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from osgeo import gdal_array
from PySignal import Signal
from skimage.transform import resize
from typing import Optional, Union

from frcpredict.util import (
    observable_property, hidden_field,
    get_canvas_params,
    gaussian_test1, doughnut_test1, airy_test1, digital_pinhole_test1
)


class PatternType(Enum):
    array2d = "array2d"
    gaussian = "gaussian"
    doughnut = "doughnut"
    airy = "airy"
    digital_pinhole = "digital_pinhole"


@dataclass
class PatternData:
    @abstractmethod
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass
class Array2DPatternData(PatternData):
    value: np.ndarray = np.zeros((81, 81))

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


@dataclass
class GaussianPatternData(PatternData):
    amplitude: float = 1.0
    fwhm: float = 480

    # Methods
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        return gaussian_test1(amplitude=self.amplitude, fwhm=self.fwhm, pixels_per_nm=pixels_per_nm)

    def __str__(self) -> str:
        return f"Gaussian; amplitude = {self.amplitude}, FWHM = {self.fwhm} nm"


@dataclass
class DoughnutPatternData(PatternData):
    periodicity: float = 540

    # Methods
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        return doughnut_test1(periodicity=self.periodicity, pixels_per_nm=pixels_per_nm)

    def __str__(self) -> str:
        return f"Doughnut; periodicity = {self.periodicity} nm"


@dataclass
class AiryPatternData(PatternData):
    amplitude: float = 1.0
    fwhm: float = 240

    # Methods
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        return airy_test1(amplitude=self.amplitude, fwhm=self.fwhm, pixels_per_nm=pixels_per_nm)

    def __str__(self) -> str:
        return f"Airy; amplitude = {self.amplitude}, FWHM = {self.fwhm} nm"


@dataclass
class DigitalPinholePatternData(PatternData):
    fwhm: float = 240

    # Methods
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        return digital_pinhole_test1(fwhm=self.fwhm, pixels_per_nm=pixels_per_nm)

    def __str__(self) -> str:
        return f"Digital pinhole; FWHM = {self.fwhm} nm"


@dataclass
class Pattern:
    pattern_type: PatternType = PatternType.array2d
    pattern_data: PatternData = Array2DPatternData()

    # Signals
    data_loaded: Signal = hidden_field(Signal)

    # Methods
    def __init__(self, pattern_type: Optional[PatternType] = None, pattern_data: Optional[PatternData] = None):
        self.data_loaded = Signal()

        if pattern_type is None and pattern_data is None:
            raise ValueError("Either pattern type or pattern data must be specified")

        if pattern_data is not None:
            self.load_data(pattern_data)
        else:
            self.load_type(pattern_type)

    def load_type(self, pattern_type: PatternType) -> None:
        if pattern_type == PatternType.array2d:
            self.pattern_data = Array2DPatternData()
        elif pattern_type == PatternType.gaussian:
            self.pattern_data = GaussianPatternData()
        elif pattern_type == PatternType.doughnut:
            self.pattern_data = DoughnutPatternData()
        elif pattern_type == PatternType.airy:
            self.pattern_data = AiryPatternData()
        elif pattern_type == PatternType.digital_pinhole:
            self.pattern_data = DigitalPinholePatternData()
        else:
            raise ValueError(f"Invalid pattern type \"{pattern_type}\"")

        self.pattern_type = pattern_type
        self.data_loaded.emit(self)

    def load_data(self, pattern_data: PatternData) -> None:
        if type(pattern_data) == Array2DPatternData:
            self.pattern_type = PatternType.array2d
        elif type(pattern_data) == GaussianPatternData:
            self.pattern_type = PatternType.gaussian
        elif type(pattern_data) == DoughnutPatternData:
            self.pattern_type = PatternType.doughnut
        elif type(pattern_data) == AiryPatternData:
            self.pattern_type = PatternType.airy
        elif type(pattern_data) == DigitalPinholePatternData:
            self.pattern_type = PatternType.digital_pinhole
        else:
            raise TypeError(f"Invalid pattern data type")

        self.pattern_data = pattern_data
        self.data_loaded.emit(self)

    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        return self.pattern_data.get_numpy_array(pixels_per_nm)

    def __str__(self) -> str:
        return str(self.pattern_data)
