########################################################################
# createRandomPartitions.py,
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
import random

class createRandomDistribution(object):

    def __init__(self, dataSet, n_splitter, pathExport):

        self.dataSet = dataSet
        self.n_splitter = n_splitter
        self.pathExport = pathExport
        self.getKeyAttribute()

    #funcion que permite obtener las listas de las columnas
    def getKeyAttribute(self):

        self.keys = []
        for key in self.dataSet:
            self.keys.append(key)

    #funcion que permite generar numero aleatorios con respecto al tamano del dataset, los numeros no se repiten
    def createRandomValuesDistribution(self):

        self.listIndex = list(range(0, len(self.dataSet)))
        random.shuffle(self.listIndex)

    #funcion que crea sub conjuntos de datos dado los valores del array, genera los archivos en formato csv de la data
    def createRandomSamples(self):
        self.createRandomValuesDistribution()

        #generamos los tamanos del conjunto de datos
        sizeData = len(self.dataSet)/self.n_splitter
        residue = len(self.dataSet)%self.n_splitter

        index = 0
        for i in range(self.n_splitter):

            name = "splitter_%d.csv"+(i+1)
            matrixData = []#almacenarara la informacion de los datos
            #generamos la matriz de datos
            for j in range(sizeData):
                rowData = []
                #agregamos el dato al conjunto de matriz
                for key in self.keys:
                    rowData.append(self.dataSet[key][self.listIndex[index]])#agregamos un elemento en la posicion index
                index+=1
                matrixData.append(rowData)

            if i == self.n_splitter-1:#debo agregar los elementos faltantes
                for j in range(len(self.dataSet)-residue, len(self.dataSet)):
                    rowData = []
                    #agregamos el dato al conjunto de matriz
                    for key in self.keys:
                        rowData.append(self.dataSet[key][self.listIndex[index]])#agregamos un elemento en la posicion index
                    index+=1
                    matrixData.append(rowData)

            #generamos el dataFrame a partir de la matriz creada
            dataFrame = pd.DataFrame(matrixData, columns=self.keys)
            dataFrame.to_csv(self.pathExport+name, index=False)
