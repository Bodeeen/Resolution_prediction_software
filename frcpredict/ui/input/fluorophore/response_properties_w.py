from frcpredict.model import IlluminationResponse
from frcpredict.ui import BaseWidget
from .response_properties_p import ResponsePropertiesPresenter


class ResponsePropertiesWidget(BaseWidget):
    """
    A widget where the user may set imaging system settings.
    """

    # Functions
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)
        self._presenter = ResponsePropertiesPresenter(self)

    def setModel(self, model: IlluminationResponse) -> None:
        self._presenter.model = model

    def updateBasicFields(self, model: IlluminationResponse) -> None:
        self.editOffToOn.setValue(model.cross_section_off_to_on)
        self.editOnToOff.setValue(model.cross_section_on_to_off)
        self.editEmission.setValue(model.cross_section_emission)
