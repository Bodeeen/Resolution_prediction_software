from abc import abstractmethod
from typing import TypeVar, Generic

from PyQt5.QtCore import QObject


Model = TypeVar('Model')


class BasePresenter(QObject, Generic[Model]):
    """
    A base class for presenters.
    """

    # Properties
    @property
    def model(self) -> Model:
        return self._model

    @model.setter
    def model(self, model: Model) -> None:
        self._model = model

    @property
    def widget(self):
        return self._widget

    # Methods
    @abstractmethod
    def __init__(self, model: Model, widget) -> None:
        self._model = model
        self._widget = widget
        super().__init__()
