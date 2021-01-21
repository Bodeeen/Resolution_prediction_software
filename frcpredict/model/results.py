from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union, Tuple, List

import numpy as np
from dataclasses_json import dataclass_json, config as json_config, Exclude
from scipy.interpolate import interp1d

from frcpredict.simulation import (
    expand_kernels_to_2d, get_frc_curve_from_kernels2d, get_expected_image_from_kernels2d
)
from frcpredict.util import ndarray_field, expand_with_multivalues
from .run_instance import RunInstance
from .sample import DisplayableSample


class KernelType(Enum):
    """
    Kernel types, corresponding to indices in the get_kernels2d function.
    """
    exp_kernel = 0  # Expected emission
    var_kernel = 1  # Variance


@dataclass_json
@dataclass
class KernelSimulationResult:
    """
    The resulting kernels from simulation. Also contains the values of the multivalue fields that
    were set when this curve was simulated; the indices in this list correspond to the indices in
    the multivalue path list in SimulationResults.
    """

    multivalue_values: List[Union[int, float]] = field(default_factory=list)

    exp_kernel: np.ndarray = ndarray_field(default_factory=lambda: np.zeros(()),
                                           encode_as_base64=True)

    var_kernel: np.ndarray = ndarray_field(default_factory=lambda: np.zeros(()),
                                           encode_as_base64=True)
    
    switches_kernel: np.ndarray = ndarray_field(default_factory=lambda: np.zeros(()),
                                           encode_as_base64=True)

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
    def resolution_at_threshold(self, run_instance: RunInstance,
                                threshold: float) -> Optional[float]:
        """
        Returns the resolution at a certain threshold, or None if the curve doesn't cross the
        threshold. run_instance must be a RunInstance without any multivalues.
        """

        x, y = self.get_frc_curve(run_instance)

        # Prevent issues when the curve crosses the threshold at multiple points, by creating a
        # copy of the y value array and modifying it so that it doesn't happen
        y = np.copy(y)
        for i in reversed(range(1, len(y))):
            if y[i] > y[i - 1]:
                y[i] = y[i - 1]

        try:
            return 1 / interp1d(y, x)(threshold)
        except ValueError:
            # Probably raised because the entered threshold is outside the interpolation range
            return None

    def get_kernels2d(self, run_instance: RunInstance, *, cache: bool = True) -> np.ndarray:
        """
        Expands the simulated kernels to 2D arrays and caches the result. Indices in the top level
        array correspond to the KernelType enum. run_instance must be a RunInstance without any
        multivalues.
        """
        if self._cached_kernels2d is not None:
            return self._cached_kernels2d
        
        kernels2d = expand_kernels_to_2d(
            self.exp_kernel, self.var_kernel, self.switches_kernel,
            canvas_inner_radius_nm=run_instance.simulation_settings.canvas_inner_radius,
            px_size_nm=run_instance.imaging_system_settings.scanning_step_size
        )

        if cache:
            self._cached_kernels2d = kernels2d

        return kernels2d

    def get_frc_curve(self, run_instance: RunInstance, *,
                      cache_kernels2d: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns a tuple containing the X and Y values respectively of the FRC curve. run_instance
        must be a RunInstance without any multivalues.
        """
        self.cache_frc_curve(run_instance, cache_kernels2d=cache_kernels2d)
        return self._cached_frc_curve_x, self._cached_frc_curve_y

    def get_expected_image(self, run_instance: RunInstance, displayable_sample: DisplayableSample,
                           *, cache_kernels2d: bool = True) -> np.ndarray:
        """
        Returns the expected image of the given sample as a 2D array. run_instance must be a
        RunInstance without any multivalues.
        """
        self.cache_expected_image(run_instance, displayable_sample, cache_kernels2d=cache_kernels2d)
        return self._cached_expected_image

    def cache_frc_curve(self, run_instance: RunInstance, *,
                        cache_kernels2d: bool = True) -> None:
        """
        Simulates the FRC curve based on the kernels and caches the result. run_instance must be a
        RunInstance without any multivalues.
        """
        if self._cached_frc_curve_x is None or self._cached_frc_curve_y is None:
            kernels2d = self.get_kernels2d(run_instance, cache=cache_kernels2d)
            self._cached_frc_curve_x, self._cached_frc_curve_y = get_frc_curve_from_kernels2d(
                kernels2d, run_instance
            )

    def cache_expected_image(self, run_instance: RunInstance, displayable_sample: DisplayableSample,
                             *, cache_kernels2d: bool = True) -> None:
        """
        Simulates the FRC curve based on the kernels and the given sample image, and caches the
        result. run_instance must be a RunInstance without any multivalues.
        """
        if (self._cached_expected_image is None
                or displayable_sample.get_id() != self._cached_expected_image_sample_id):
            kernels2d = self.get_kernels2d(run_instance, cache=cache_kernels2d)
            self._cached_expected_image = get_expected_image_from_kernels2d(
                kernels2d, run_instance, displayable_sample
            )
            self._cached_expected_image_sample_id = displayable_sample.get_id()

    def clear_cache(self, *, clear_kernels2d: bool = False, clear_frc_curves: bool = False,
                    clear_expected_image: bool = False) -> None:
        """ Clears the specified caches. """

        if clear_kernels2d:
            self._cached_kernels2d = None

        if clear_frc_curves:
            self._cached_frc_curve_x = None
            self._cached_frc_curve_y = None

        if clear_expected_image:
            self._cached_expected_image = None
            self._cached_expected_image_sample_id = None


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
        default_factory=lambda: np.zeros(()),
        custom_encoder=lambda arr: arr,
        custom_decoder=lambda arr: np.frompyfunc(KernelSimulationResult.from_dict, 1, 1)(np.array(arr))
    )

    # Methods
    def precache(self, *, cache_kernels2d: bool = False, cache_frc_curves: bool = False,
                 cache_expected_image_for: Optional[DisplayableSample] = None) -> None:
        """ Pre-caches the chosen elements in all kernel result instances. """

        for kernel_result in np.nditer(self.kernel_results, flags=["refs_ok"]):
            kernel_result_item = kernel_result.item()

            run_instance = expand_with_multivalues(self.run_instance,
                                                   self.multivalue_paths,
                                                   kernel_result_item.multivalue_values)

            has_expected_image = cache_expected_image_for is not None

            if cache_frc_curves:
                kernel_result_item.cache_frc_curve(run_instance, cache_kernels2d=has_expected_image)

            if has_expected_image:
                kernel_result_item.cache_expected_image(run_instance, cache_expected_image_for)

            if not cache_kernels2d:
                kernel_result_item.clear_cache(clear_kernels2d=True)

    def clear_cache(self, *, clear_kernels2d: bool = False, clear_frc_curves: bool = False,
                    clear_expected_image: bool = False) -> None:
        """ Clears the chosen caches in all kernel result instances. """
        for kernel_result in np.nditer(self.kernel_results, flags=["refs_ok"]):
            kernel_result_item = kernel_result.item()
            kernel_result_item.clear_cache(clear_kernels2d=clear_kernels2d,
                                           clear_frc_curves=clear_frc_curves,
                                           clear_expected_image=clear_expected_image)
