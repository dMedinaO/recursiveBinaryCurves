########################################################################
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

import numpy as np
import pandas as pd

class statisticsSummary(object):

    def __init__(self, dataSet, pathResponse):

        self.dataSet = dataSet
        self.pathResponse = pathResponse

        self.createMatrixResponse()#generamos el archivo de salida con la data de interes

    #metodo que permite calcular los estadisticos para una columna en el set de datos...
    def calculateValuesForColumn(self, attribute):

        dictResponse = []
        dictResponse.append(attribute)#feature
        dictResponse.append(np.mean(self.dataSet[attribute]))#mean
        dictResponse.append(np.std(self.dataSet[attribute]))#std
        dictResponse.append(np.var(self.dataSet[attribute]))#var
        dictResponse.append(max(self.dataSet[attribute]))#min
        dictResponse.append(min(self.dataSet[attribute]))#max

        return dictResponse

    #metodo que permite formar la matriz para obtener la data de estadisticas
    def createMatrixResponse(self):

        matrixData = []

        for key in self.dataSet:

            matrixData.append(self.calculateValuesForColumn(key))

        dfExport = pd.DataFrame(matrixData, columns=["Feature", "Mean", "StandarDeviation", "Variance", "MaxValue", "MinValue"])
        dfExport.to_csv(self.pathResponse+"statisticsSummary.csv", index=False)
