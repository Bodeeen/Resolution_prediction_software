from typing import Union, List

from frcpredict.model import SimulationResults
from frcpredict.util import get_dataclass_field_description


def getLabelForMultivalue(results: SimulationResults, multivaluePath: List[Union[int, str]]) -> str:
    """
    Returns the name that should be displayed for the field with the specified multivalue path.
    This works by concatenating the descriptions of the segments of the multivalue path.
    """

    label = ""
    containerDataclass = results.run_instance

    multivaluePathIndex = 0
    while multivaluePathIndex < len(multivaluePath):
        multivaluePathSegment = multivaluePath[multivaluePathIndex]

        isLastSegment = multivaluePathIndex >= len(multivaluePath) - 1
        isList = not isLastSegment and isinstance(multivaluePath[multivaluePathIndex + 1], int)

        dataclassField = containerDataclass.__dataclass_fields__[multivaluePathSegment]

        if dataclassField.name not in ["fluorophore_settings", "imaging_system_settings",
                                       "pulse_scheme", "sample_properties", "detector_properties",
                                       "simulation_settings",
                                       "pattern_data"]:  # Don't include these in the name
            if len(label) > 0:
                label += " "

            segmentDescription, _ = get_dataclass_field_description(
                containerDataclass, dataclassField,
                list_or_dict_index=-1 if not isList else multivaluePath[multivaluePathIndex + 1]
            )
            label += segmentDescription

        if not isLastSegment:
            containerDataclass = getattr(containerDataclass, multivaluePathSegment)
            if isList:
                containerDataclass = containerDataclass[multivaluePath[multivaluePathIndex + 1]]

        multivaluePathIndex += 1 if not isList else 2

    label = label[:1].upper() + label[1:]  # Convert first character to uppercase
    return label
