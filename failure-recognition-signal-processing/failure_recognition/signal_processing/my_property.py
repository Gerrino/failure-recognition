"""Module providing the MyProperty class"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, List, TypeVar
from unicodedata import name
from failure_recognition.signal_processing import (
    DEFAULT_FLOAT,
    DEFAULT_INT,
    DEFAULT_MAX_FLOAT,
    DEFAULT_MAX_INT,
    DEFAULT_MIN_FLOAT,
    DEFAULT_MIN_INT,
)


@dataclass
class MyProperty:
    """Class providing property information used by tsfresh feature

    Example
    -------

    MyProperty class, e.g. lag: Name="lag", Type.SystemType=int,
    Properties: Name, Type
    """

    name: str
    type: MyType
    id_prefix: str = ""
    enabled: bool = True

    @classmethod
    def from_json(cls, json_obj: dict, id_prefix: str = "") -> MyProperty:
        json_obj["id_prefix"] = id_prefix
        my_property = cls(**json_obj)
        my_property.type = MyType.from_json(json_obj["type"], my_property.id)
        return my_property

    @property
    def id(self) -> str:    # e.g. agg_autocorrelation_param_f_agg
        return f"{self.id_prefix}_{self.name}"

    def get_default_value(self):
        """Get the default value of the property depending on the type"""
        if self.type.system_type == "int":
            return DEFAULT_INT if self.type.default_value is None else self.type.default_value
        if self.type.system_type == "float":
            return DEFAULT_FLOAT if self.type.default_value is None else self.type.default_value
        if self.type.system_type == "string":
            return self.type.range[0]

    def get_range(self, i: int = None) -> list:
        """Get this properties range itmes"""
        out_range = self.type.range
        if len(self.type.range) == 2 and self.type.range[0] == 0 and self.type.range[1] == 0:
            if self.type.system_type == "int":
                out_range = [DEFAULT_MIN_INT, DEFAULT_MAX_INT]
            if self.type.system_type == "float":
                out_range = [DEFAULT_MIN_FLOAT, DEFAULT_MAX_FLOAT]
        if i >= 0:
            return out_range[i]
        return out_range

    def get_key_value_pair(self, cfg: dict, sensor) -> dict:
        """Get name/value dictionary of this property. if cfg contains a parameter value for
        this property and sensor return the value from the cfg.
        """
        properties_of_dict = {}
        if self.type.system_type == "dictionary":
            for p in self.type.property_list:
                properties_of_dict.update(p.get_key_value_pair(cfg, sensor))
            return {self.name: properties_of_dict}
        value = None
        if cfg != None and self.get_id(sensor) in cfg:
            value = cfg[self.get_id(sensor)]
        else:
            value = self.get_default_value()
        return {self.name: value}

    def get_hyper_parameter_list(self, sensor) -> list:
        import ConfigSpace.hyperparameters as smac_params

        default_val = self.get_default_value()
        if self.type.system_type == "int":
            return [
                smac_params.UniformIntegerHyperparameter(
                    self.get_id(sensor),
                    self.get_range(0),
                    self.get_range(1),
                    default_value=default_val,
                )
            ]
        if self.type.system_type == "string":
            return [smac_params.CategoricalHyperparameter(self.get_id(sensor), self.type.range, self.type.range[0])]
        if self.type.system_type == "float":
            return [
                smac_params.UniformFloatHyperparameter(
                    self.get_id(sensor),
                    self.get_range(0),
                    self.get_range(1),
                    default_value=default_val,
                )
            ]
        if self.type.system_type == "bool":
            return [smac_params.UniformIntegerHyperparameter(self.get_id(sensor), 0, 1, default_value=0)]
        if self.type.system_type == "dictionary":
            hyper_parameters_of_dict = []
            for p in self.type.property_list:
                hyper_parameters_of_dict.append(
                    p.get_hyper_parameter_list(sensor)[0])
            return hyper_parameters_of_dict

    def get_id(self, sensor):
        return sensor + "__" + self.id


T = TypeVar('T')


@dataclass
class MyType(Generic[T]):
    """Class providing type information for a MyProperty

    Example
    -------
    MyType class, e.g. string with SystemType="string" and range "Range" of optional values this type can assume
    """
    system_type: str
    range: List[T] = None
    default_value: T = None
    property_list: List[MyProperty] = None
    element_type: str = None
    array_of_my_type: bool = None

    @classmethod
    def from_json(cls, json_obj, id_prefix) -> MyType:
        my_type = cls(**json_obj)
        if my_type.system_type == "dictionary":
            my_type.property_list = []
            for property_item in json_obj["property_list"]:
                my_type.property_list.append(
                    MyProperty.from_json(property_item, id_prefix))
        return my_type


if __name__ == "__main__":
    type = MyType.from_json({"range": [5], "system_type": "int",
                             "default_value": 5}, "yolo_")
    type.default_value = 5

    pass
