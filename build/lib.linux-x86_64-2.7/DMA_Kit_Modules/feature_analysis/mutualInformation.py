# coding=utf-8
########################################################################
# mutualInformation.py,
#
# Execute Mutual Information feature analysis building a matrix between features.
# Ends with values 0 (independents) or 1 (same)
#Receives a dataset without labels of clustering.
#
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
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import normalized_mutual_info_score as mis
import numpy as np
import pandas as pd

#metodos de la libreria utils...
from DMA_Kit_Modules.utils import transformDataClass
from DMA_Kit_Modules.utils import transformFrequence
from DMA_Kit_Modules.utils import ScaleNormalScore
from DMA_Kit_Modules.utils import ScaleMinMax
from DMA_Kit_Modules.utils import ScaleDataSetLog
from DMA_Kit_Modules.utils import ScaleLogNormalScore
from DMA_Kit_Modules.utils import encodingFeatures

class mutualInformation(object):

    def __init__(self, dataSet, pathResponse, optionNormalize):

        self.pathResponse = pathResponse
        self.dataSet = dataSet
        self.optionNormalize = optionNormalize
        ###########

    #metodo que permite normalizar el set de datos con respecto a la opcion entregada
    def normalizeDataSet(self):

        #ahora transformamos el set de datos por si existen elementos discretos...
        encoding = encodingFeatures.encodingFeatures(self.dataSet, 20)
        encoding.evaluEncoderKind()
        dataSetNewFreq = encoding.dataSet

        dataSetNorm = ""
        #ahora aplicamos el procesamiento segun lo expuesto
        if self.optionNormalize == 1:#normal scale
            applyNormal = ScaleNormalScore.applyNormalScale(dataSetNewFreq)
            dataSetNorm = applyNormal.dataTransform

        if self.optionNormalize == 2:#min max scaler
            applyMinMax = ScaleMinMax.applyMinMaxScaler(dataSetNewFreq)
            dataSetNorm = applyMinMax.dataTransform

        if self.optionNormalize == 3:#log scale
            applyLog = ScaleDataSetLog.applyLogScale(dataSetNewFreq)
            dataSetNorm = applyLog.dataTransform

        if self.optionNormalize == 4:#log normal scale
            applyLogNormal = ScaleLogNormalScore.applyLogNormalScale(dataSetNewFreq)
            dataSetNorm = applyLogNormal.dataTransform

        return dataSetNorm

    def singleMI(self, array1, array2):
        mi = mis(array1,array2)
        return mi
        ###########

    #funcion que permite obtener los datos del data set a partir de una key
    def getDataFromDataSet(self, key, dataSet):

        row = []

        for i in range(len(dataSet)):
            row.append(dataSet[key][i])
        return row

    def makeMatrix(self):

        # type_c puede ser
        # arithmetic, min, max, geometric
        # metodo de normalizar el denominador
        okiedoki=""
        type_c="arithmetic"

        try:
            data= self.normalizeDataSet()

            columnas = data.columns.tolist()

            W = np.empty((len(columnas), len(columnas)))# se crea 1 matriz vacia
            i=0
            j=0
            for key1 in columnas:
                for key2 in columnas:
                    A = self.getDataFromDataSet(key1, data)
                    B = self.getDataFromDataSet(key2, data)
                    mi = self.singleMI(A,B)

                    W[i][j] = mi
                    j+=1
                i+=1
                j=0

            #CSV
            file = "%sMatrixMI.csv" % (self.pathResponse)
            df = pd.DataFrame(W, index=columnas, columns=columnas)
            df.to_csv(file)

            #generamos la imagen
            plt.figure()
            heatmap = sns.heatmap(df)

            loc, labels = plt.xticks()
            heatmap.set_xticklabels(labels)
            heatmap.set_yticklabels(labels[::-1])
            nameFileImage = "%smutualInformationMatrix.svg" % (self.pathResponse)
            plt.savefig(nameFileImage)

            okiedoki = "OK"

        except Exception as e:
            #raise e
            okiedoki = "ERROR"
            pass
        return okiedoki
        ###########
