from .colour_utils import wavelength_to_rgb

from .dataclass_extras import dataclass_internal_attrs, observable_property, clear_signals

from .pattern_utils import (
    get_canvas_params,
    gaussian_test1, doughnut_test1, airy_test1, digital_pinhole_test1, physical_pinhole_test1
)

from .range_utils import (
    rangeable_field, get_paths_of_ranges, expand_ranges, avg_value_if_range, get_value_from_path
)
