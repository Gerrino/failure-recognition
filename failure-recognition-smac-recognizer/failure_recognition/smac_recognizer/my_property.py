"""Moudle providing the MyProperty class"""

from typing import List
from failure_recognition.smac_recognizer import (
    DEFAULT_FLOAT,
    DEFAULT_INT,
    DEFAULT_MAX_FLOAT,
    DEFAULT_MAX_INT,
    DEFAULT_MIN_FLOAT,
    DEFAULT_MIN_INT,
)


class MyProperty:
    """Class providing property information used by tsfresh feature

    Example
    -------

    MyProperty class, e.g. lag: Name="lag", Type.SystemType=int,
    Properties: Name, Type
    """

    name: str

    def __init__(self, json_obj, id_prefix):
        self.__dict__ = json_obj
        self.id = f"{id_prefix}_{self.name}"  # e.g. agg_autocorrelation_param_f_agg
        self.type = MyType(json_obj["type"], self.id)

    def get_default_value(self):
        """Get the default value of the property depending on the type"""
        if self.type.system_type == "int":
            return getattr(self.type, "default_value", DEFAULT_INT)
        if self.type.system_type == "float":
            return getattr(self.type, "default_value", DEFAULT_FLOAT)
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
        properties_of_dict = {}
        if self.type.system_type == "dictionary":
            for p in self.type.property_list:
                properties_of_dict.update(p.get_key_value_pair(cfg, sensor))
            return properties_of_dict
        if cfg != None:
            return {self.name: cfg[self.get_id(sensor)]}
        else:
            return {self.name: self.get_default_value()}

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
                hyper_parameters_of_dict.append(p.get_hyper_parameter_list(sensor)[0])
            return hyper_parameters_of_dict

    def get_id(self, sensor):
        return sensor + "__" + self.id


class MyType:
    """Class providing type information for a MyProperty

    Example
    -------
    MyType class, e.g. string with SystemType="string" and range "Range" of optional values this type can assume
    """

    property_list: List[MyProperty]

    def __init__(self, json_obj, id_prefix):
        self.system_type = json_obj["system_type"]
        if self.system_type == "dictionary":
            self.property_list = []
            for obj in json_obj["property_list"]:
                self.property_list.append(MyProperty(obj, id_prefix))
        if "range" in json_obj:
            self.range = json_obj["range"]
        if "default_value" in json_obj:
            self.default_value = json_obj["default_value"]
