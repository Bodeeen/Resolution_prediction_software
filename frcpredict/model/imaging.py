from dataclasses import dataclass, field
import numpy as np
from osgeo import gdal_array
from PySignal import Signal


@dataclass
class ImagingSystemSettings:
    optical_psf: np.ndarray
    pinhole_function: np.ndarray
    scanning_step_size: float

    # Internal fields
    _scanning_step_size: float = field(init=False, repr=False, default=0.0)
    _initialized: bool = field(init=False, repr=False, default=False)  # TODO: Fix this ugly stuff

    # Properties
    @property
    def optical_psf(self) -> np.ndarray:
        return self._optical_psf

    @optical_psf.setter
    def optical_psf(self, optical_psf: np.ndarray) -> None:
        self._optical_psf = optical_psf
        if self._initialized:
            self.optical_psf_changed.emit(optical_psf)

    @property
    def pinhole_function(self) -> np.ndarray:
        return self._pinhole_function

    @pinhole_function.setter
    def pinhole_function(self, pinhole_function: np.ndarray) -> None:
        self._pinhole_function = pinhole_function
        if self._initialized:
            self.pinhole_function_changed.emit(pinhole_function)

    @property
    def scanning_step_size(self) -> float:
        return self._scanning_step_size

    @scanning_step_size.setter
    def scanning_step_size(self, scanning_step_size: float) -> None:
        self._scanning_step_size = scanning_step_size
        if self._initialized:
            self.basic_field_changed.emit(self)

    # Functions
    def __post_init__(self):  # TODO: Fix this ugly stuff
        self.optical_psf_changed = Signal()
        self.pinhole_function_changed = Signal()
        self.basic_field_changed = Signal()
        self._initialized = True

    def load_optical_psf_npy(self, path: str) -> None:
        """ Loads an optical PSF from an .npy file. """
        self.optical_psf = np.load(path)

    def load_optical_psf_image(self, path: str) -> None:
        """ Loads an optical PSF from an image file. """
        self.optical_psf = self._get_numpy_array_from_image(path)

    def load_pinhole_function_npy(self, path: str) -> None:
        """ Loads a pinhole function from an .npy file. """
        self.pinhole_function = np.load(path)

    def load_pinhole_function_image(self, path: str) -> None:
        """ Loads an pinhole function from an image file. """
        self.pinhole_function = self._get_numpy_array_from_image(path)

    # Internal functions
    def _get_numpy_array_from_image(self, path: str) -> np.ndarray:
        """ Converts an image to a numpy array. """

        raster_array = gdal_array.LoadFile(path)  # Load image as numpy array

        if np.issubdtype(raster_array.dtype, np.integer):
            raster_array = raster_array / 256.0  # Image has int values, normalise

        raster_array = np.mean(raster_array, axis=0)  # Average values over channels

        return raster_array
