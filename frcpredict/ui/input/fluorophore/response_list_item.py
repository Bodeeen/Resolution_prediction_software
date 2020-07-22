from PyQt5.QtGui import QColor, QPixmap, QIcon
from PyQt5.QtWidgets import QListWidgetItem

from frcpredict.model import IlluminationResponse
from frcpredict.ui import ListItemWithValue
from frcpredict.util import wavelength_to_rgb


class ResponseListItem(ListItemWithValue):
    """
    Custom QListWidgetItem that can be initialized/updated with a fluorophore response and sorted by
    wavelength, and displays an colour indicator that indicates the wavelength.
    """

    # Properties
    @property
    def wavelength(self) -> float:
        return self._wavelength

    # Methods
    def __init__(self, response: IlluminationResponse, *args, **kwargs) -> None:
        self._wavelength = response.wavelength

        super().__init__(text=str(response), value=response.wavelength, *args, **kwargs)
        self.setIcon(self._generateIcon(response.wavelength))

    # Internal methods
    def _generateIcon(self, wavelength: float) -> QIcon:
        pixmap = QPixmap(12, 32)
        pixmap.fill(QColor(*wavelength_to_rgb(wavelength), 255))
        return QIcon(pixmap)
