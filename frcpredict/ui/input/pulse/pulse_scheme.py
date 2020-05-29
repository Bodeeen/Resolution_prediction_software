from frcpredict.model import PulseScheme, Pulse
from frcpredict.ui import BaseWidget


class PulseSchemeWidget(BaseWidget):
    """
    A widget where the user may set the pulse scheme.
    """

    # Functions
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

        self.model = PulseScheme(
            pulses=[]
        )
