from dataclasses import dataclass, field
from dataclasses_json import config, Exclude
import numpy as np
from osgeo import gdal_array
from PySignal import Signal

from frcpredict.util import observable_field


@dataclass
class ImagingSystemSettings:
    optical_psf: np.ndarray = observable_field(
        "_optical_psf", default=None,
        signal_name="optical_psf_changed", emit_arg_name="optical_psf"
    )

    pinhole_function: np.ndarray = observable_field(
        "_pinhole_function", default=None,
        signal_name="pinhole_function_changed", emit_arg_name="pinhole_function")

    scanning_step_size: float = observable_field(
        "_scanning_step_size", default=0.0,
        signal_name="basic_field_changed"
    )

    # Signals
    optical_psf_changed: Signal = field(
        init=False, repr=False, default_factory=Signal, metadata=config(exclude=Exclude.ALWAYS))
    pinhole_function_changed: Signal = field(
        init=False, repr=False, default_factory=Signal, metadata=config(exclude=Exclude.ALWAYS))
    basic_field_changed: Signal = field(
        init=False, repr=False, default_factory=Signal, metadata=config(exclude=Exclude.ALWAYS))

    # Functions
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
