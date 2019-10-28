########################################################################
# launcherScanPrediction.py,
#
# script that allows to execute all the algorithms with different variations to generate
# classifiers and their respective models, stores in a csv file the data that is generated with
# regarding the various measures that are obtained as a result of the process ...
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

from modulesNLM.utils import transformDataClass
from modulesNLM.utils import transformFrequence
from modulesNLM.utils import ScaleNormalScore
from modulesNLM.utils import ScaleMinMax
from modulesNLM.utils import ScaleDataSetLog
from modulesNLM.utils import ScaleLogNormalScore
from modulesNLM.utils import summaryScanProcess
from modulesNLM.utils import responseResults
from modulesNLM.utils import encodingFeatures

#para evaluar la performance
from modulesNLM.supervised_learning_predicction import performanceData

#para generar la regression
from sklearn import linear_model
import pandas as pd

#clase con la responsabilidad de evaluar si el conjunto de datos para entrenamiento de regresion es lineal o no, con respecto a
#la respuesta de interes
class checkNonLinearRegression(object):

    def __init__(self, dataSet, featureClass, threshold):

        self.dataSet = dataSet
        self.featureClass = featureClass
        self.threshold = threshold

    #preparamos el conjunto de datos
    def prepareDataSet(self):

        self.dataResponse = self.dataSet[self.featureClass]#obtenemos la variable respuesta

        dictData = {}

        for key in self.dataSet:
            if key != self.featureClass:
                arrayFeature = []
                for i in self.dataSet[key]:
                    arrayFeature.append(i)
                dictData.update({key:arrayFeature})

        #formamos el nuevo set de datos...
        dataSetParser = pd.DataFrame(dictData)

        #codificacion del conjunto de datos
        encoding = encodingFeatures.encodingFeatures(dataSetParser, 20)
        encoding.evaluEncoderKind()
        dataSetNewFreq = encoding.dataSet
        #ahora aplicamos el procesamiento segun lo expuesto
        applyNormal = ScaleNormalScore.applyNormalScale(dataSetNewFreq)
        self.data = applyNormal.dataTransform

    #metodo que permite hacer el entrenamiento del modelo...
    def applyLinearRegression(self):

        try:
            self.reg = linear_model.LinearRegression()
            self.reg.fit(self.data, self.dataResponse)

            score = self.reg.score(self.data, self.dataResponse)
            print "Performance score: ", score
            if score<= self.threshold:
                return 0#no lineal
            else:
                return 1#lineal
        except:
            return -1#esto indica un error
            pass
