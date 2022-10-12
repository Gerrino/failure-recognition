"""Module providing the feature class"""
from __future__ import annotations
from dataclasses import dataclass, field
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
    coefficients: MyProperty = None
    

    @classmethod
    def from_json(cls, json_obj: dict) -> Feature:
        feature = cls(**json_obj)
        feature.input_parameters = [MyProperty.from_json(p, feature.name) for p in list(feature.input_parameters)]
        if feature.coefficients is not None:
            feature.coefficients = MyProperty.from_json(feature.coefficients, feature.name)
        return feature

    def __str__(self):
        return f"Feature '{self.name}'"

    def __repr__(self):
        return f"Feature '{self.name}'"

    def get_parameter_dict(self, cfg, sensor) -> dict:
        """Get the parameter dict of the feature with all 
        key-value pairs of the input parameters, e.g.
        {"input_var_0": designated_value}
        
        """
        parameter_dict = {}
        for input_param in self.input_parameters:
            parameter_dict.update(input_param.get_key_value_pair(cfg, sensor))
        return parameter_dict
