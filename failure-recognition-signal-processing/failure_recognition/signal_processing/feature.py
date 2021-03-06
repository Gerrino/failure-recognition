"""Module providing the feature class"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
from failure_recognition.signal_processing.my_property import MyProperty


@dataclass
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
    return_type: str

    @classmethod
    def from_json(cls, json_obj: dict) -> Feature:
        feature = cls(**json_obj)
        tmp_input_params = []
        for obj in feature.input_parameters:
            input_param = MyProperty.from_json(obj, feature.name)
            tmp_input_params.append(input_param)
        feature.input_parameters = tmp_input_params
        return feature

    def __str__(self):
        return f"Feature '{self.name}'"

    def __repr__(self):
        return f"Feature '{self.name}'"

    def get_parameter_dict(self, cfg, sensor) -> dict:
        parameter_dict = {}
        for input_param in self.input_parameters:
            parameter_dict.update(input_param.get_key_value_pair(cfg, sensor))
        return parameter_dict
