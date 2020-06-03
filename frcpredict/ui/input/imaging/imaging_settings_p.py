import numpy as np

from PyQt5.QtCore import pyqtSlot, QObject
from PyQt5.QtWidgets import QFileDialog

from frcpredict.model import ImagingSystemSettings
from frcpredict.ui import BaseWidget
from frcpredict.ui.util import getArrayPixmap


class ImagingSystemSettingsPresenter(QObject):
    """
    Presenter for the imaging system settings widget.
    """

    # Properties
    @property
    def model(self) -> ImagingSystemSettings:
        return self._model

    @model.setter
    def model(self, model: ImagingSystemSettings) -> None:
        self._model = model

        # Update data in widget
        self._onOpticalPsfChange(model.optical_psf)
        self._onPinholeFunctionChange(model.pinhole_function)
        self._onBasicFieldChange(model)

        # Prepare model events
        model.optical_psf_changed.connect(self._onOpticalPsfChange)
        model.pinhole_function_changed.connect(self._onPinholeFunctionChange)
        model.basic_field_changed.connect(self._onBasicFieldChange)

    # Methods
    def __init__(self, widget, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._widget = widget

        # Prepare UI events
        self._widget.btnLoadOpticalPsf.clicked.connect(self._onClickLoadOpticalPsf)
        self._widget.btnLoadPinholeFunction.clicked.connect(self._onClickLoadPinholeFunction)

        # Initialize model
        self.model = ImagingSystemSettings(
            optical_psf=np.zeros((80, 80)),
            pinhole_function=np.zeros((80, 80)),
            scanning_step_size=1.0
        )

    # Model event handling
    def _onOpticalPsfChange(self, opticalPsf: np.ndarray) -> None:
        """ Loads the optical PSF into a visualization in the interface. """
        self._widget.setOpticalPsfPixmap(getArrayPixmap(opticalPsf))

    def _onPinholeFunctionChange(self, pinholeFunction: np.ndarray) -> None:
        """ Loads the pinhole function into a visualization in the interface. """
        self._widget.setPinholeFunctionPixmap(getArrayPixmap(pinholeFunction))

    def _onBasicFieldChange(self, model: ImagingSystemSettings) -> None:
        """ Loads basic model fields (spinboxes etc.) into the interface fields. """
        self._widget.updateBasicFields(model)

    # UI event handling
    @pyqtSlot()
    def _onClickLoadOpticalPsf(self) -> None:
        """ Lets the user open a file that contains optical PSF data, and loads the file. """

        path, _ = QFileDialog.getOpenFileName(
            self._widget, "Open optical PSF file", filter="Supported files (*.tif *.tiff *.png *.npy)")

        if path:
            if path.endswith(".npy"):
                self.model.load_optical_psf_npy(path)
            else:
                self.model.load_optical_psf_image(path)

    @pyqtSlot()
    def _onClickLoadPinholeFunction(self) -> None:
        """ Lets the user open a file that contains pinhole function data, and loads the file. """

        path, _ = QFileDialog.getOpenFileName(
            self._widget, "Open pinhole function file", filter="Supported files (*.tif *.tiff *.png *.npy)")

        if path:
            if path.endswith(".npy"):
                self.model.load_pinhole_function_npy(path)
            else:
                self.model.load_pinhole_function_image(path)
