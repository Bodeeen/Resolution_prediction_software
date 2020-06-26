from PyQt5.QtWidgets import QListWidgetItem

from frcpredict.model import IlluminationResponse


class ResponseListItem(QListWidgetItem):
    """
    Custom QListWidgetItem that can be initialized/updated with and sorted by wavelength.
    """

    # Properties
    @property
    def wavelengthStart(self) -> int:
        return self._wavelengthStart

    @property
    def wavelengthEnd(self) -> int:
        return self._wavelengthEnd

    # Methods
    def __init__(self, response: IlluminationResponse, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._wavelengthStart = response.wavelength_start
        self._wavelengthEnd = response.wavelength_end
        self.setText(str(response))

    def __lt__(self, other: QListWidgetItem) -> bool:
        if self.wavelengthStart != other.wavelengthStart:
            return self.wavelengthStart < other.wavelengthStart
        else:
            return self.wavelengthEnd < other.wavelengthEnd

    def __gt__(self, other: QListWidgetItem) -> bool:
        if self.wavelengthStart != other.wavelengthStart:
            return self.wavelengthStart > other.wavelengthStart
        else:
            return self.wavelengthEnd > other.wavelengthEnd
