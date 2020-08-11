from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, List

import numpy as np
from dataclasses_json import dataclass_json, config as json_config, Exclude
from scipy.interpolate import interp1d

from frcpredict.simulation import (
    expand_kernels_to_2d, get_frc_curve_from_kernels2d, get_expected_image_from_kernels2d
)
from frcpredict.util import ndarray_field, expand_with_multivalues
from .run_instance import RunInstance


@dataclass_json
@dataclass
class KernelSimulationResult:
    """
    The resulting kernels from simulation. Also contains the values of the multivalue fields that
    were set when this curve was simulated; the indices in this list correspond to the indices in
    the multivalue path list in SimulationResults.
    """

    multivalue_values: List[Union[int, float]] = field(default_factory=list)

    exp_kernel: np.ndarray = ndarray_field(default=np.zeros(()), encode_as_base64=True)

    var_kernel: np.ndarray = ndarray_field(default=np.zeros(()), encode_as_base64=True)

    _cached_kernels2d: Optional[np.ndarray] = ndarray_field(default=None, encode_as_base64=True,
                                                            exclude=Exclude.ALWAYS)

    _cached_frc_curve_x: Optional[np.ndarray] = ndarray_field(default=None, encode_as_base64=True,
                                                              exclude=Exclude.ALWAYS)

    _cached_frc_curve_y: Optional[np.ndarray] = ndarray_field(default=None, encode_as_base64=True,
                                                              exclude=Exclude.ALWAYS)

    _cached_expected_image: Optional[np.ndarray] = ndarray_field(default=None, encode_as_base64=True,
                                                                 exclude=Exclude.ALWAYS)

    _cached_expected_image_sample_id: Optional[str] = field(default=None,
                                                            metadata=json_config(exclude=Exclude.ALWAYS))

    # Methods
    def resolution_at_threshold(self, run_instance: RunInstance, threshold: float) -> Optional[float]:
        """
        Returns the resolution at a certain threshold, or None if the curve doesn't cross the
        threshold.
        """

        x, y = self.get_frc_curve(run_instance)

        # Prevent issues when the curve crosses the threshold at multiple points, by creating a
        # copy of the y value array and modifying it so that it doesn't happen
        y = np.copy(y)
        for i in range(1, len(y)):
            if y[i] > y[i - 1]:
                y[i] = y[i - 1]

        try:
            return 1 / interp1d(y, x)(threshold)
        except ValueError:
            # Probably raised because the entered threshold is outside the interpolation range
            return None

    def get_frc_curve(self, run_instance: RunInstance) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns a tuple containing the X and Y values respectively of the FRC curve. """
        self.cache_frc_curve(run_instance)
        return self._cached_frc_curve_x, self._cached_frc_curve_y

    def get_expected_image(self, run_instance: RunInstance, sample_id: str,
                           sample_image: np.ndarray) -> np.ndarray:
        """ Returns the expected image of the given sample as a 2D array. """
        self.cache_expected_image(run_instance, sample_id, sample_image)
        return self._cached_expected_image

    def cache_frc_curve(self, run_instance: RunInstance) -> None:
        """ Simulates the FRC curve based on the kernels and caches the result. """
        if self._cached_frc_curve_x is None or self._cached_frc_curve_y is None:
            self._cache_kernels2d(run_instance)
            self._cached_frc_curve_x, self._cached_frc_curve_y = get_frc_curve_from_kernels2d(
                self._cached_kernels2d, run_instance
            )
            self._post_caching_cleanup()

    def cache_expected_image(self, run_instance: RunInstance, sample_id: str,
                             sample_image_arr: np.ndarray) -> None:
        """
        Simulates the FRC curve based on the kernels and the given sample image, and caches the
        result.
        """
        if (self._cached_expected_image is None
                or sample_id != self._cached_expected_image_sample_id):
            self._cache_kernels2d(run_instance)
            self._cached_expected_image = get_expected_image_from_kernels2d(
                self._cached_kernels2d, sample_image_arr
            )
            self._cached_expected_image_sample_id = sample_id
            self._post_caching_cleanup()

    # Internal methods
    def _cache_kernels2d(self, run_instance: RunInstance) -> None:
        """ Expands the simulated kernels to 2D arrays and caches the result. """
        if self._cached_kernels2d is None:
            self._cached_kernels2d = expand_kernels_to_2d(
                self.exp_kernel, self.var_kernel,
                pixels_per_nm=run_instance.imaging_system_settings.scanning_step_size
            )

    def _post_caching_cleanup(self) -> None:
        """ Cleans up memory if possible, intended to be run after caching. """
        if (self._cached_frc_curve_x is not None and self._cached_frc_curve_y is not None
                and self._cached_expected_image is not None):
            self._cached_kernels2d = None


@dataclass_json
@dataclass
class SimulationResults:
    """
    Results of a simulation. This includes the RunInstance that was used as input, a list of
    paths to all multivalues in the RunInstance, and an array of kernel simulation results that were
    output from the simulation.

    If the RunInstance contains multivalues, the kernel result array will be of the shape
    (n_1, n_2, n_3, ...), where n_1 is the number of states the first multivalue can be in, n_2, is
    the number of states the second multivalue can be in, and so on. Otherwise, the kernel result
    array will be a 1-dimensional array with one item.
    """

    run_instance: RunInstance = field(default_factory=RunInstance)

    multivalue_paths: List[List[Union[int, str]]] = field(default_factory=list)

    kernel_results: np.ndarray = ndarray_field(  # array of KernelResult
        default=np.zeros(()),
        custom_encoder=lambda arr: arr,
        custom_decoder=lambda arr: np.frompyfunc(KernelSimulationResult.from_dict, 1, 1)(np.array(arr))
    )

    # Methods
    def cache_all(self, sample_id: Optional[str], sample_image_arr: Optional[np.ndarray]) -> None:
        """ Caches all FRC curves and expected images. """

        for kernel_result in np.nditer(self.kernel_results, flags=["refs_ok"]):
            kernel_result_item = kernel_result.item()

            run_instance = expand_with_multivalues(self.run_instance,
                                                   self.multivalue_paths,
                                                   kernel_result_item.multivalue_values)

            kernel_result_item.cache_frc_curve(run_instance)

            if sample_id is not None and sample_image_arr is not None:
                kernel_result_item.cache_expected_image(run_instance, sample_id, sample_image_arr)
