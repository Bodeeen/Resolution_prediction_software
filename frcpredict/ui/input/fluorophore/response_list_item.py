from PyQt5.QtWidgets import QListWidgetItem

from frcpredict.model import IlluminationResponse


class ResponseListItem(QListWidgetItem):
    """
    Custom QListWidgetItem that can be initialized/updated with and sorted by wavelength.
    """

    # Properties
    @property
    def wavelength(self) -> int:
        return self._wavelength

    # Methods
    def __init__(self, response: IlluminationResponse, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._wavelength = response.wavelength
        self.setText(str(response))

    def __lt__(self, other: QListWidgetItem) -> bool:
        return self.wavelength < other.wavelength

    def __gt__(self, other: QListWidgetItem) -> bool:
        return self.wavelength > other.wavelength
