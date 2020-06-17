from frcpredict.ui import BaseWidget


class PresetPickerWidget(BaseWidget):
    """
    A widget where the user may load, save, import, and export presets.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)
