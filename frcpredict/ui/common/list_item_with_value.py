from typing import Optional, TypeVar, Generic

from PyQt5.QtWidgets import QListWidgetItem

T = TypeVar("T")
U = TypeVar("U")


class ListItemWithValue(QListWidgetItem, Generic[T, U]):
    """
    Custom QListWidgetItem that holds a value and can be sorted by it.
    """

    # Methods
    def __init__(self, text: str, value: T, tag: Optional[U] = None, *args, **kwargs) -> None:
        super().__init__(text, *args, **kwargs)
        self._value = value
        self._tag = tag

    def value(self) -> T:
        return self._value

    def tag(self) -> Optional[U]:
        return self._tag

    def __lt__(self, other: QListWidgetItem) -> bool:
        if isinstance(self.value(), (int, float)):
            return self.value() < other.value()
        else:
            return super().__lt__(other)

    def __gt__(self, other: QListWidgetItem) -> bool:
        if isinstance(self.value(), (int, float)):
            return self.value() > other.value()
        else:
            return super().__lt__(other)
