from frcpredict.model import CameraProperties
from frcpredict.ui import BaseWidget


class CameraPropertiesWidget(BaseWidget):
    """
    A widget where the user may set camera properties.
    """

    # Functions
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

        self.model = CameraProperties(
            read_out_noise=0.0,
            quantum_efficiency=0.0
        )
