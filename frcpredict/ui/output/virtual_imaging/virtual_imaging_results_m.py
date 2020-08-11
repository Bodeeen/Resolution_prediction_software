from dataclasses import dataclass


@dataclass
class VirtualImagingResultsModel:
    """
    Model for the virtual imaging results widget.
    """

    panZoomAutoReset: bool = False
    autoLevelAutoPerform: bool = True
    autoLevelLowerCutoff: float = 0.0
