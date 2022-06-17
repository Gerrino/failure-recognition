#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 12:00:27 2021

@author: gerritnoske
"""
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

#MyProperty class, e.g. lag: Name="lag", Type.SystemType=int,
#Properties: Name, Type
class MyProperty:
    DEFAULT_MAX_INT = 100
    DEFAULT_MIN_INT = 0
    DEFAULT_MAX_FLOAT = 10
    DEFAULT_MIN_FLOAT = -10
    def __init__(self, jsonObj, idPrefix):        
        self.__dict__ = jsonObj;
        self.ID = f"{idPrefix}_{self.Name}" # e.g. agg_autocorrelation_param_f_agg
        self.Type = MyType(jsonObj["Type"], self.ID)
    def GetDefaultValue(self):
        defaultInt = 1
        defaultFloat = 0.1
        if hasattr(self.Type, "DefaultValue"):
            defaultInt = self.Type.DefaultValue
            defaultFloat = self.Type.DefaultValue
        if self.Type.SystemType == "int":
            return defaultInt
        if self.Type.SystemType == "float":
            return defaultFloat
        if self.Type.SystemType == "string":
            return self.Type.Range[0]
    def GetRange(self, i = None) -> list:
        outRange = self.Type.Range
        if len(self.Type.Range) == 2 and self.Type.Range[0] == 0 and self.Type.Range[1] == 0:
            if self.Type.SystemType == "int":
                outRange = [self.DEFAULT_MIN_INT, self.DEFAULT_MAX_INT]
            if self.Type.SystemType == "float":
                outRange = [self.DEFAULT_MIN_FLOAT, self.DEFAULT_MAX_FLOAT]
        if i >= 0:
            return outRange[i]
        return outRange
    def GetKeyValuePair(self, cfg, sensor) -> dict:
        propertiesOfDict = {}
        if self.Type.SystemType == "dictionary":            
            for p in self.Type.PropertyList:
                propertiesOfDict.update(p.GetKeyValuePair(cfg, sensor))
            return propertiesOfDict
        if cfg != None:
            return {self.Name : cfg[self.GetID(sensor)]}
        else:
            return {self.Name : self.GetDefaultValue()}
    def GetHyperParameterList(self, sensor) -> list:
        default_val = self.GetDefaultValue()
        if self.Type.SystemType == "int":            
            return [UniformIntegerHyperparameter(self.GetID(sensor), self.GetRange(0), self.GetRange(1), default_value=default_val)]
        if self.Type.SystemType == "string":
            return [CategoricalHyperparameter(self.GetID(sensor), self.Type.Range, self.Type.Range[0])]
        if self.Type.SystemType == "float":
            return [UniformFloatHyperparameter(self.GetID(sensor), self.GetRange(0), self.GetRange(1), default_value=default_val)]
        if self.Type.SystemType == "bool":
            return [UniformIntegerHyperparameter(self.GetID(sensor), 0, 1, default_value=0)]
        if self.Type.SystemType == "dictionary":
            hyperParametersOfDict = []
            for p in self.Type.PropertyList:
                hyperParametersOfDict.append(p.GetHyperParameterList(sensor)[0])
            return hyperParametersOfDict
        
    def GetID(self, sensor):
        return sensor + "__" + self.ID
        
#MyType class, e.g. string with SystemType="string" and range "Range" of optional values this type can assume
class MyType:
    def __init__(self, jsonObj, idPrefix):
        self.SystemType = jsonObj["SystemType"]        
        if self.SystemType == 'dictionary':
            self.PropertyList = []
            for obj in jsonObj["PropertyList"]:                
                self.PropertyList.append(MyProperty(obj, idPrefix))
        if "Range" in jsonObj:
            self.Range = jsonObj["Range"]
        if "DefaultValue" in jsonObj:
            self.DefaultValue = jsonObj["DefaultValue"]