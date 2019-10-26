########################################################################
# ScaleMinMax.py,
#
# Normalize data set with min max pattern.
#
# Copyright (C) 2019  David Medina Ortiz, david.medina@cebib.cl
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
########################################################################

import pandas as pd
import numpy as np

class applyMinMaxScaler(object):

    def __init__(self, dataSet):

        self.dataSet = dataSet

        dicValues = {}
        for key in self.dataSet:
            row = self.transformNormalScale(key)

            #create a data frame
            dicValues.update({key:row})
        self.dataTransform = pd.DataFrame(dicValues)

    #metodo que permite definir el tipo de data...
    def checkData(self, array):
        response = 0

        try:
            for i in array:
                data = float(i)
        except:
            response = 1
        return response

    #metodo que toma una columna del set de datos y aplica la transformacion logaritmica de los datos...
    def transformNormalScale(self, feature):

        dataValue = []

        if self.checkData(self.dataSet[feature]) == 0:

            minValue = min(self.dataSet[feature])
            maxValue = max(self.dataSet[feature])

            for i in self.dataSet[feature]:

                valueData = float(i)
                value = (valueData-minValue)/(maxValue-minValue)
                dataValue.append(value)
        else:
            for i in self.dataSet[feature]:
                dataValue.append(i)
        return dataValue
