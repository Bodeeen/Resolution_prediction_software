from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from dataclasses_json import dataclass_json
from skimage.io import imread
from skimage.transform import resize

from frcpredict.util import (
    get_canvas_params, extended_field,
    generate_gaussian, generate_doughnut, generate_airy,
    generate_digital_pinhole, generate_physical_pinhole,
    ndarray_field
)
from .multivalue import Multivalue


@dataclass_json
@dataclass
class PatternData(ABC):
    """
    Abstract class that describes the actual properties of a pattern.
    """

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
    """
    A description of a pattern given by a 2D array, typically loaded from an image.
    """

    value: np.ndarray = ndarray_field(default=np.zeros((1, 1)))

    # Methods
    def get_numpy_array(self, pixels_per_nm: Optional[float] = None) -> np.ndarray:
        """
        Returns a numpy array representation of the 2D array. If pixels_per_nm value is set, the 2D
        array will be resized under the assumption that it has an inner radius of the same size as
        the constant _canvas_inner_radius_nm.
        """

        if pixels_per_nm is not None:
            _, canvas_side_length_px = get_canvas_params(pixels_per_nm)
            canvas_size = (canvas_side_length_px, canvas_side_length_px)

            if self.value.shape != canvas_size:
                # TODO: This assumes that the loaded pattern has an inner radius of the same length
                #       as the constant _canvas_inner_radius_nm; this may not always be correct
                return resize(self.value, canvas_size, order=3)

        return self.value

    def is_empty(self) -> bool:
        return not np.any(self.value)

    def __str__(self) -> str:
        if not self.is_empty():
            return "Loaded from file"
        else:  # All zeros in value
            return "Empty pattern"

    @classmethod
    def from_npy_file(cls, path: str) -> "Array2DPatternData":
        """
        Loads a 2D array from an .npy file. The file is expected to contain a float array that is
        of the shape (width, height).
        """
        return cls(value=np.load(path))

    @classmethod
    def from_image_file(cls, path: str) -> "Array2DPatternData":
        """ Loads a 2D array from an image file. """
        raster_array = imread(path, as_gray=True)  # Load image as numpy array

        if np.issubdtype(raster_array.dtype, np.integer) and raster_array.max() > 1:
            raster_array = raster_array / 255.0  # Image has int values; normalize

        return cls(value=raster_array)


@dataclass_json
@dataclass
class GaussianPatternData(PatternData):
    """
    A description of the properties of a gaussian pattern.
    """

    amplitude: Union[float, Multivalue[float]] = extended_field(1.0, description="amplitude")
    fwhm: Union[float, Multivalue[float]] = extended_field(480.0, description="FWHM [nm]")

    # Methods
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        return generate_gaussian(amplitude=self.amplitude, fwhm=self.fwhm,
                                 pixels_per_nm=pixels_per_nm)

    def __str__(self) -> str:
        return f"Gaussian; amplitude = {self.amplitude}, FWHM = {self.fwhm} nm"


@dataclass_json
@dataclass
class DoughnutPatternData(PatternData):
    """
    A description of the properties of a doughnut pattern.
    """

    periodicity: Union[float, Multivalue[float]] = extended_field(540.0,
                                                                  description="periodicity [nm]")

    # Methods
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        return generate_doughnut(periodicity=self.periodicity, pixels_per_nm=pixels_per_nm)

    def __str__(self) -> str:
        return f"Doughnut; periodicity = {self.periodicity} nm"


@dataclass_json
@dataclass
class AiryFWHMPatternData(PatternData):
    """
    A description of the properties of an airy pattern.
    """

    amplitude: Union[float, Multivalue[float]] = extended_field(1.0, description="amplitude")
    fwhm: Union[float, Multivalue[float]] = extended_field(240.0, description="FWHM [nm]")

    # Methods
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        return generate_airy(amplitude=self.amplitude, fwhm=self.fwhm, pixels_per_nm=pixels_per_nm)

    def __str__(self) -> str:
        return f"Airy; amplitude = {self.amplitude}, FWHM = {self.fwhm} nm"


@dataclass_json
@dataclass
class AiryNAPatternData(PatternData):
    """
    A description of the properties of an airy pattern.
    """

    na: Union[float, Multivalue[float]] = extended_field(0.8, description="NA")
    emission_wavelength: Union[float, Multivalue[float]] = extended_field(
        250.0, description="em. wavelength [nm]"
    )

    # Methods
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        return generate_airy(amplitude=1.0,
                             fwhm=self.emission_wavelength / (2 * self.na),
                             pixels_per_nm=pixels_per_nm)

    def __str__(self) -> str:
        return f"Airy; NA = {self.na}, emission wavelength = {self.emission_wavelength} nm"


@dataclass_json
@dataclass
class DigitalPinholePatternData(PatternData):
    """
    A description of the properties of a digital pinhole pattern.
    """

    fwhm: Union[float, Multivalue[float]] = 240.0  # nanometres

    # Methods
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        return generate_digital_pinhole(fwhm=self.fwhm, pixels_per_nm=pixels_per_nm)

    def __str__(self) -> str:
        return f"Digital pinhole; FWHM = {self.fwhm} nm"


@dataclass_json
@dataclass
class PhysicalPinholePatternData(PatternData):
    """
    A description of the properties of a physical pinhole pattern.
    """

    radius: Union[float, Multivalue[float]] = extended_field(100.0, description="radius [nm]")

    # Methods
    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        return generate_physical_pinhole(radius=self.radius, pixels_per_nm=pixels_per_nm)

    def __str__(self) -> str:
        return f"Physical pinhole; radius = {self.radius} nm"