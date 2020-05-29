from dataclasses import dataclass

from .fluorophore import FluorophoreSettings
from .imaging import ImagingSystemSettings
from .pulse import PulseScheme
from .sample import SampleProperties
from .camera import CameraProperties


@dataclass
class RunInstance:
    fluorophore_settings: FluorophoreSettings
    imaging_system_settings: ImagingSystemSettings
    pulse_scheme: PulseScheme
    sample_properties: SampleProperties
    camera_properties: CameraProperties
