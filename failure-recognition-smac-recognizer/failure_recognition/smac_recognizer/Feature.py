"""Module providing the feature class"""

from typing import List
from MyProperty import MyProperty


class Feature:
    """Feature class

    Examples
    --------

    Properties: Name, InputParameters
    InputParameters is a list with element type: MyProperty
    """

    Enabled: bool
    Name: str
    InputParameters: List[MyProperty]

    def __init__(self, jsonObj):
        self.__dict__ = jsonObj
        tmpInputParams = []
        for obj in self.InputParameters:
            inputParam = MyProperty(obj, self.Name)
            tmpInputParams.append(inputParam)
        self.InputParameters = tmpInputParams

    def __str__(self):
        return f"Feature '{self.Name}'"

    def __repr__(self):
        return f"Feature '{self.Name}'"

    def GetParameterDict(self, cfg, sensor) -> dict:
        parameterDict = {}
        for inputParam in self.InputParameters:
            parameterDict.update(inputParam.GetKeyValuePair(cfg, sensor))
        return parameterDict
