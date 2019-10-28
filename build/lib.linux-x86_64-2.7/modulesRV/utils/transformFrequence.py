########################################################################
# transformFrequence.py,
#
# Transform data discrete to continues in frequence space.
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

class frequenceData(object):

    def __init__(self, dataSet):

        self.dataSet = dataSet

        dicValues = {}
        for key in self.dataSet:
            row = self.processFrequence(key)

            #create a data frame
            dicValues.update({key:row})

        self.dataTransform = pd.DataFrame(dicValues)

    #metodo que transforma la clase en valores continuos desde el 0 hasta el n...
    def transformClassData(self, array):

        #preguntamos si la clase es letras o numeros
        if self.checkData(array) == 1:
            valueClass = list(set(array))

            dictResponse = {}
            index=0
            for value in valueClass:
                dictResponse.update({value:index})
                index+=1

            dataResponse = []

            for i in array:
                dataResponse.append(dictResponse[i])
            return dataResponse
        else:
            dataResponse = array
            return dataResponse

    #metodo que permite definir el tipo de data...
    def checkData(self, array):
        response = 0

        try:
            for i in array:
                data = float(i)
        except:
            response = 1
        return response

    def processFrequence(self, feature):

        dataValue = []
        if self.checkData(self.dataSet[feature]) == 1:

            dictFrequence = self.getFrequence(self.dataSet[feature], feature)
            print dictFrequence

            #ahora hacemos la transformacion de los elementos
            for i in self.dataSet[feature]:
                dataValue.append(dictFrequence[i])
        else:
            dataValue = self.dataSet[feature]

        return dataValue

    #metodo que permite formar las frecuencias
    def getFrequence(self, arrayData, nameFeature):

        #hacemos el conteo...
        data = list(set(arrayData))

        dictResponse = {}

        for element in data:
            cont=0

            for value in arrayData:
                if element == value:
                    cont+=1
            freq = float(cont)/float(len(arrayData))

            dictResponse.update({element:freq})

        return dictResponse
