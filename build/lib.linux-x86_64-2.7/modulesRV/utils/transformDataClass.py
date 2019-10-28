########################################################################
# transformDataClass.py,
#
# Change values of features response with discrete distribution to continue distribution in frequence space
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

class transformClass(object):

    def __init__(self, arrayData):

        self.arrayData = arrayData
        self.transformData, self.dictTransform = self.transformClassData(self.arrayData)

    #metodo que permite definir el tipo de data...
    def checkData(self, array):
        response = 0

        try:
            for i in array:
                data = float(i)
        except:
            response = 1
        return response

    #metodo que transforma la clase en valores continuos desde el 0 hasta el n...
    def transformClassData(self, array):

        #preguntamos si la clase es letras o numeros
        valueClass = list(set(array))
        dictResponse = {}
        dataResponse = []

        if self.checkData(array) == 1:
            index=0
            for value in valueClass:
                dictResponse.update({value:index})
                index+=1
            for i in array:
                dataResponse.append(dictResponse[i])
        else:
            dataResponse = array

        return dataResponse, dictResponse
