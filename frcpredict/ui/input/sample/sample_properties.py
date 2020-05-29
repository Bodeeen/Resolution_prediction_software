from frcpredict.model import SampleProperties
from frcpredict.ui import BaseWidget


class SamplePropertiesWidget(BaseWidget):
    """
    A widget where the user may set sample properties.
    """

    # Functions
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

        self.model = SampleProperties(
            spectral_power=0.0,
            labelling_density=0.0
        )
