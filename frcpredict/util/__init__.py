from .colour_utils import wavelength_to_rgb

from .dataclass_extras import (
    dataclass_internal_attrs, dataclass_with_observables, observable_field, clear_signals
)

from .pattern_utils import (
    get_canvas_params,
    generate_gaussian, generate_doughnut, generate_airy,
    generate_digital_pinhole, generate_physical_pinhole
)

from .multivalue_utils import (
    multi_accepting_field, get_paths_of_multivalues, expand_multivalues, avg_value_if_multivalue,
    get_value_from_path
)
