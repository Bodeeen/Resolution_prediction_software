from PyQt5.QtGui import QColor, QPixmap, QIcon
from PyQt5.QtWidgets import QListWidgetItem

from frcpredict.model import IlluminationResponse
from frcpredict.util import wavelength_to_rgb


class ResponseListItem(QListWidgetItem):
    """
    Custom QListWidgetItem that can be initialized/updated with and sorted by wavelength.
    """

    # Properties
    @property
    def wavelength(self) -> float:
        return self._wavelength

    # Methods
    def __init__(self, response: IlluminationResponse, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._wavelength = response.wavelength
        self.setIcon(self._generateIcon(response.wavelength))
        self.setText(str(response))

    def __lt__(self, other: QListWidgetItem) -> bool:
        return self.wavelength < other.wavelength

    def __gt__(self, other: QListWidgetItem) -> bool:
        return self.wavelength > other.wavelength

    # Internal methods
    def _generateIcon(self, wavelength: float) -> QIcon:
        pixmap = QPixmap(12, 32)
        pixmap.fill(QColor(*wavelength_to_rgb(wavelength), 255))
        return QIcon(pixmap)
