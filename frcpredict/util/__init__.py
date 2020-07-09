from .colour_utils import wavelength_to_rgb

from .dataclass_extras import dataclass_internal_attrs, observable_property, clear_signals

from .pattern_utils import (
    get_canvas_params,
    gaussian_test1, doughnut_test1, airy_test1, digital_pinhole_test1, physical_pinhole_test1
)

from .multivalue_utils import (
    multi_accepting_field, get_paths_of_multivalues, expand_multivalues, avg_value_if_multivalue,
    get_value_from_path
)
