#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 10:11:25 2021

@author: gerritnoske
"""

from MyProperty import MyProperty
            
#Properties: Name, InputParameters
#InputParameters is a list with element type: MyProperty
class Feature:
    """Feature class"""
    Enabled: bool
    Name: str

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




       
            
        