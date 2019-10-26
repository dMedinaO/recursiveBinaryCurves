########################################################################
# createConfusionMatrix.py,
#
# Create a confusion matrix for a given model
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

import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix

from DMA_Kit_Modules.graphic import createCharts

import itertools
import pandas as pd
import json

class confusionMatrix(object):

    def __init__(self, dataSet, target, modelData, cv_values, path, classList):

        self.dataSet = dataSet
        self.target = target
        self.modelData = modelData
        self.path = path
        self.classList = classList
        self.pathResponse = self.path+"confusionMatrix"

        if cv_values == -1:
            self.cv_values = LeaveOneOut()
            self.cv_valuesName = "LeaveOneOut"
        else:
            self.cv_values = cv_values

    #metodo que permite exportar la matriz a un csv y adiciona las filas y columnas correspondientes a fiabilidad y bakanosidad
    def exportConfusionMatrix(self, matrix, dictTransform):

        #calculamos la sensitividad del modelo (en base a los valores de la primera columna)
        bakanosidad = []
        for i in range(len(matrix)):
            sumRow = sum(matrix[i])
            value = (matrix[i][i]/float(sumRow))*100
            bakanosidad.append(value)

        #calculamos la especificidad del modelo...
        transpose = matrix.transpose()
        fiabilidad = []
        for i in range(len(transpose)):
            sumRow = sum(transpose[i])
            value = (transpose[i][i]/float(sumRow))*100
            fiabilidad.append(value)


        header = []
        for element in self.classList:
            header.append(self.getKeyToDict(dictTransform, element))

        matrixData = []
        for element in matrix:#obtenemos las columnas
            rowSum = sum(element)
            row = []
            for value in element:
                dataInValue = (value/float(rowSum))*100
                row.append(dataInValue)
            matrixData.append(row)
        dictResponse = {"Specificity": fiabilidad, "Sensitivity":bakanosidad, "matrix":matrixData, "header": header}
        #generamos el grafico de barras comparativas entre estas medidas
        graph = createCharts.graphicsCreator()

        graph.createBarChartCompare(fiabilidad, bakanosidad, 'Specificity', 'Sensitivity', 'Class Response', 'Percentage', "Quality of the model", self.target, self.path+"barchartCompare.png")

        return dictResponse

    #metodo que retorna la key de un diccionario dado su valor
    def getKeyToDict(self, dictTransform, value):

        keyValue=""
        for key in dictTransform:
            if dictTransform[key] == value:
                keyValue= key
                break
        return keyValue

    #metodo que permite generar la matriz de confusion...
    def createConfusionMatrix(self, dictTransform):

        self.predictions = cross_val_predict(self.modelData, self.dataSet, self.target, cv=self.cv_values)
        matrix = confusion_matrix(self.target, self.predictions)
        dictResponse = self.exportConfusionMatrix(matrix, dictTransform)
        graph = createCharts.graphicsCreator()

        graph.createConfusionMatrixPictures(matrix, self.target, self.path+"confusionMatrix.svg")

        return dictResponse

        # Plot non-normalized confusion matrix
        #plt.figure()
        #self.plot_confusion_matrix(matrix, classes=self.classList, title='Confusion matrix, without normalization')
        #plt.savefig(self.pathResponse)
