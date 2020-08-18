from copy import deepcopy

from PyQt5.QtCore import pyqtSlot

from frcpredict.ui import BasePresenter, Preferences


class PreferencesPresenter(BasePresenter[Preferences]):
    """
    Presenter for the preferences dialog.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: Preferences) -> None:
        # Disconnect old model event handling
        try:
            self._model.basic_field_changed.disconnect(self._onBasicFieldChange)
        except AttributeError:
            pass

        # Set model
        self._model = model

        # Trigger model change event handlers
        self._onBasicFieldChange(model)

        # Prepare model events
        model.basicFieldChanged.connect(self._onBasicFieldChange)

    # Methods
    def __init__(self, widget) -> None:
        super().__init__(deepcopy(Preferences.get()), widget)

        # Prepare UI events
        widget.precacheFrcCurvesChanged.connect(self._uiPrecacheFrcCurveChange)
        widget.precacheExpectedImagesChanged.connect(self._uiPrecacheExpectedImagesChanged)
        widget.cacheKernels2DChanged.connect(self._uiCacheKernels2DChanged)

    # Model event handling
    def _onBasicFieldChange(self, model: Preferences) -> None:
        """ Loads basic model fields (e.g. ints) into the widget. """
        self.widget.updateBasicFields(model)

    # UI event handling
    @pyqtSlot(int)
    def _uiPrecacheFrcCurveChange(self, value: bool) -> None:
        self.model.precacheFrcCurves = value

    @pyqtSlot(int)
    def _uiPrecacheExpectedImagesChanged(self, value: bool) -> None:
        self.model.precacheExpectedImages = value

    @pyqtSlot(int)
    def _uiCacheKernels2DChanged(self, value: bool) -> None:
        self.model.cacheKernels2D = value
