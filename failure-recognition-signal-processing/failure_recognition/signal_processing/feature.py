"""Module providing the feature class"""

from typing import List
from failure_recognition.signal_processing.my_property import MyProperty


class Feature:
    """Feature class

    Examples
    --------

    properties: name, input_parameters
    input_parameters is a list with element type: MyProperty
    """

    enabled: bool
    name: str
    input_parameters: List[MyProperty]

    def __init__(self, json_obj):
        self.__dict__ = json_obj
        tmp_input_params = []
        for obj in self.input_parameters:
            input_param = MyProperty(obj, self.name)
            tmp_input_params.append(input_param)
        self.input_parameters = tmp_input_params

    def __str__(self):
        return f"Feature '{self.name}'"

    def __repr__(self):
        return f"Feature '{self.name}'"

    def get_parameter_dict(self, cfg, sensor) -> dict:
        parameter_dict = {}
        for input_param in self.input_parameters:
            parameter_dict.update(input_param.get_key_value_pair(cfg, sensor))
        return parameter_dict
