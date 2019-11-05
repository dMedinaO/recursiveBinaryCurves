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

import pandas as pd
import os
from modulesRV.clustering_analysis import evaluationClustering

class evaluatedClusteringProcess(object):

    def __init__(self, pathResult, header):

        self.pathResult = pathResult
        self.header = header

        self.getNameFilesInResponse()
        self.createUniqDataSet()#archivo unico para formar el conjunto de datos
        self.getPerformance()#obtener las medidas de desempeno

    #metodo que permite poder obtener
    def getNameFilesInResponse(self):

        listFiles = os.listdir(self.pathResult)

        #obtenemos solo los archivos en formato *.csv
        self.listCluster = []
        for element in listFiles:
            if ".csv" in element:
                self.listCluster.append(element)

    #metodo que permite formar el unico conjunto de datos para hacer la evaluacion de desempeno
    def createUniqDataSet(self):
        self.matrixData = []
        self.header = []
        self.arrayClass = []

        #obtenemos el header
        inputFile = pd.read_csv(self.pathResult+self.listCluster[0])
        for element in inputFile:
            self.header.append(element)

        indexClass = 0

        #generamos una matriz para formas las descripciones de los grupos obtenidos
        self.matrixDescription = []

        #obtenemos la data y las clases
        for element in self.listCluster:

            readDoc = pd.read_csv(self.pathResult+element)
            for i in range(len(readDoc)):
                rowData = []
                for value in self.header:
                    rowData.append(readDoc[value][i])
                self.matrixData.append(rowData)
                self.arrayClass.append(indexClass)
            indexClass+=1

            row = []
            id = element.split(".")[0]
            members = len(readDoc)
            row.append(id)
            row.append(members)
            self.matrixDescription.append(row)

        #exportamos el dataFrame
        dfExport = pd.DataFrame(self.matrixDescription, columns=['ID', 'Members'])
        dfExport.to_csv(self.pathResult+"summaryGroups.csv", index=False)

    #metodo que permite hacer la evaluacion de la medida de desempeno
    def getPerformance(self):

        evaluateClustering = evaluationClustering.evaluationClustering(self.matrixData, self.arrayClass)
        self.calinski = evaluateClustering.calinski
        self.siluetas = evaluateClustering.siluetas
