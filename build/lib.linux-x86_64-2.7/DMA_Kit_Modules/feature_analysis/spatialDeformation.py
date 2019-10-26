########################################################################
# spatialDeformation.py,
#
# Generates spacial deformation of elements.
# Generate ranking of features using random forest approach
# If dataset has no labels, they are calculated using clustering methods, then they are chosen using calinsky and silhouette approachs.
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

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
import numpy as np

#metodos de la libreria utils...
from DMA_Kit_Modules.utils import transformDataClass
from DMA_Kit_Modules.utils import transformFrequence
from DMA_Kit_Modules.utils import ScaleNormalScore
from DMA_Kit_Modules.utils import ScaleMinMax
from DMA_Kit_Modules.utils import ScaleDataSetLog
from DMA_Kit_Modules.utils import ScaleLogNormalScore
from DMA_Kit_Modules.graphic import createCharts
from DMA_Kit_Modules.utils import encodingFeatures

class spatialDeformation(object):

    def __init__(self, dataSet, pathResponse, optionNormalize):

        self.dataSet = dataSet
        self.pathResponse = pathResponse
        self.optionNormalize = optionNormalize

    #metodo que permite normalizar el set de datos con respecto a la opcion entregada
    def normalizeDataSet(self, dataSetOri):

        #ahora transformamos el set de datos por si existen elementos discretos...
        encoding = encodingFeatures.encodingFeatures(dataSetOri, 20)
        encoding.evaluEncoderKind()
        dataSetNewFreq = encoding.dataSet

        dataSetNorm = ""
        #ahora aplicamos el procesamiento segun lo expuesto
        if self.optionNormalize == 1:#normal scale
            applyNormal = ScaleNormalScore.applyNormalScale(dataSetNewFreq)
            dataSetNorm = applyNormal.dataTransform

        elif self.optionNormalize == 2:#min max scaler
            applyMinMax = ScaleMinMax.applyMinMaxScaler(dataSetNewFreq)
            dataSetNorm = applyMinMax.dataTransform

        elif self.optionNormalize == 3:#log scale
            applyLog = ScaleDataSetLog.applyLogScale(dataSetNewFreq)
            dataSetNorm = applyLog.dataTransform

        else:#log normal scale
            applyLogNormal = ScaleLogNormalScore.applyLogNormalScale(dataSetNewFreq)
            dataSetNorm = applyLogNormal.dataTransform

        return dataSetNorm

    #metodo que permite aplicar random Forest para obtener las importancias...
    def applyRandomForestClassifier(self, data, target):

        response = ""

        try:
        #instancia a random forest y aplicacion del mismo
            random_forest = RandomForestClassifier(max_depth=2, random_state=0, n_estimators=10, n_jobs=-1, criterion='gini')
            random_forest = random_forest.fit(data, target)

            #obtenemos las importancias
            importances = pd.DataFrame({'feature':data.columns.tolist(),'importance':np.round(random_forest.feature_importances_,3)})
            importances = importances.sort_values('importance',ascending=False).set_index('feature')

            #exportamos el resultado
            nameCSV = "%srankingImportance.csv" % (self.pathResponse)
            importances.to_csv(nameCSV)

            #generamos el grafico de las relevancias
            dataP = pd.read_csv(nameCSV)
            keys = dataP['feature']
            values = dataP['importance']
            for i in range(len(values)):
                values[i] = values[i]*100

            namePicture = self.pathResponse+"RelevanceRanking_SpatialCLF.png"
            #instanciamos un objeto del tipo graph
            graph = createCharts.graphicsCreator()
            graph.createBarChart(keys, values, 'Component', 'Relevance (%)', 'Ranking Relevance Components', namePicture)

            response = "OK"
        except:
            response = "ERROR"
            pass
        return response

    #metodo que permite aplicar random Forest para obtener las importancias...
    def applyRandomForestPrediction(self, data, target):

        response = ""
        try:
            #instancia a random forest y aplicacion del mismo
            random_forest = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=10, n_jobs=-1, criterion='mse')
            random_forest = random_forest.fit(data, target)

            #obtenemos las importancias
            importances = pd.DataFrame({'feature':data.columns.tolist(),'importance':np.round(random_forest.feature_importances_,3)})
            importances = importances.sort_values('importance',ascending=False).set_index('feature')

            #exportamos el resultado
            nameCSV = "%srankingImportance.csv" % (self.pathResponse)
            importances.to_csv(nameCSV)

            #generamos el grafico de las relevancias
            dataP = pd.read_csv(nameCSV)
            keys = dataP['feature']
            values = dataP['importance']
            for i in range(len(values)):
                values[i] = values[i]*100
            namePicture = self.pathResponse+"RelevanceRanking_SpatialPRD.png"

            #instanciamos un objeto del tipo graph
            graph = createCharts.graphicsCreator()
            graph.createBarChart(keys, values, 'Component', 'Relevance (%)', 'Ranking Relevance Components', namePicture)

            response = "OK"
        except:
            response = "ERROR"
            pass

        return response
    #metodo que permite obtener las clases y los atributos desde un set de datos
    def getClass_Attribute(self, dataSet, featureResponse):

        targetResponse = self.dataSet[featureResponse]
        dictData = {}

        for key in self.dataSet:
            if key != featureResponse:
                arrayFeature = []
                for i in self.dataSet[key]:
                    arrayFeature.append(i)
                dictData.update({key:arrayFeature})

        #formamos el nuevo set de datos...
        dataSetNew = pd.DataFrame(dictData)

        return dataSetNew, targetResponse

    #metodo que permite aplicar la deformacion de espacio...
    def applySpatialDeformation(self, feature, kindDataSet):

        #try:
        if kindDataSet == 'CLASS':
            data, target = self.getClass_Attribute(self.dataSet, feature)

            #normalizo el set de datos...
            dataNorm = self.normalizeDataSet(data)

            #transformamos las clases en variables numericas si es necesario...
            transformData = transformDataClass.transformClass(target)
            targetTransform = transformData.transformData

            response = self.applyRandomForestClassifier(dataNorm, targetTransform)

        elif kindDataSet == 'PREDICTION':
            data, response = self.getClass_Attribute(self.dataSet, feature)

            #normalizamos el set de datos...
            dataNorm = self.normalizeDataSet(data)
            response = self.applyRandomForestPrediction(dataNorm, response)

        else:
            response = "Option not available for this type of data set"
        #except:
        #    response = "ERROR"
        #    pass
        return response
