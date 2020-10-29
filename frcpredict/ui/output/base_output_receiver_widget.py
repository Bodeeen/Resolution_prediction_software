from PyQt5.QtCore import pyqtSignal

from frcpredict.ui import BaseWidget
from .controls.output_director_m import ViewOptions
from .controls.output_director_w import OutputDirectorWidget


class BaseOutputReceiverWidget(BaseWidget):
    # Signals
    kernelResultChanged = pyqtSignal(object, object, bool)
    viewOptionsChanged = pyqtSignal(ViewOptions)

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        self._outputDirector = None
        super().__init__(*args, **kwargs)

    def outputDirector(self) -> OutputDirectorWidget:
        """ Returns the output director that this widget will receive its data from. """
        return self._outputDirector

    def setOutputDirector(self, outputDirector: OutputDirectorWidget) -> None:
        """ Sets the output director that this widget will receive its data from. """

        if self._outputDirector is not None:
            self._outputDirector.kernelResultChanged.disconnect(self.kernelResultChanged)
            self._outputDirector.viewOptionsChanged.disconnect(self.viewOptionsChanged)

        if outputDirector is not None:
            outputDirector.kernelResultChanged.connect(self.kernelResultChanged)
            outputDirector.viewOptionsChanged.connect(self.viewOptionsChanged)

        self._outputDirector = outputDirector
