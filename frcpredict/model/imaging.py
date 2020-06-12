from dataclasses import dataclass, field
from dataclasses_json import config as json_config, Exclude
import numpy as np
from osgeo import gdal_array
from PySignal import Signal

from frcpredict.util import observable_property, hidden_field


@dataclass
class ImagingSystemSettings:
    optical_psf: np.ndarray = observable_property(
        "_optical_psf", default=np.zeros((80, 80)),
        signal_name="optical_psf_changed", emit_arg_name="optical_psf"
    )

    pinhole_function: np.ndarray = observable_property(
        "_pinhole_function", default=np.zeros((80, 80)),
        signal_name="pinhole_function_changed", emit_arg_name="pinhole_function")

    scanning_step_size: float = observable_property(
        "_scanning_step_size", default=0.0,
        signal_name="basic_field_changed"
    )

    # Signals
    optical_psf_changed: Signal = hidden_field(Signal)
    pinhole_function_changed: Signal = hidden_field(Signal)
    basic_field_changed: Signal = hidden_field(Signal)

    # Methods
    def load_optical_psf_npy(self, path: str) -> None:
        """ Loads an optical PSF from an .npy file. The file is expected to contain a float
        array that is of the shape (width, height) and has values within the range [0, 1]. """
        self.optical_psf = np.load(path)

    def load_optical_psf_image(self, path: str) -> None:
        """ Loads an optical PSF from an image file. """
        self.optical_psf = self._get_numpy_array_from_image(path)

    def load_pinhole_function_npy(self, path: str) -> None:
        """ Loads a pinhole function from an .npy file. The file is expected to contain a float
        array that is of the shape (width, height) and has values within the range [0, 1]. """
        self.pinhole_function = np.load(path)

    def load_pinhole_function_image(self, path: str) -> None:
        """ Loads an pinhole function from an image file. """
        self.pinhole_function = self._get_numpy_array_from_image(path)

    # Internal methods
    def _get_numpy_array_from_image(self, path: str) -> np.ndarray:
        """ Converts an image to a numpy array. """

        raster_array = gdal_array.LoadFile(path)  # Load image as numpy array

        if np.issubdtype(raster_array.dtype, np.integer):
            raster_array = raster_array / 255.0  # Image has int values; normalise

        if raster_array.ndim > 2:
            # Image has multiple channels; average the values over the channels
            raster_array = np.mean(raster_array, axis=0)

        return raster_array
