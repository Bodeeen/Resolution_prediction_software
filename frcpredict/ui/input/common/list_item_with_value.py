from typing import TypeVar, Generic

from PyQt5.QtWidgets import QListWidgetItem


T = TypeVar('T')


class ListItemWithValue(QListWidgetItem, Generic[T]):
    """
    Custom QListWidgetItem that can be initialized/updated with and sorted by wavelength.
    """

    # Methods
    def __init__(self, text: str, value: T, *args, **kwargs) -> None:
        super().__init__(text, *args, **kwargs)
        self._value = value

    def value(self) -> T:
        return self._value
