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

from modulesNLM.supervised_learning_analysis import NuSVM
from modulesNLM.supervised_learning_analysis import SVM
from modulesNLM.statistics_analysis import summaryStatistic

import pandas as pd

class checkNonLinearClass(object):

    def __init__(self, dataSet, featureClass, threshold):

        self.dataSet = dataSet
        self.featureClass = featureClass
        self.threshold = threshold

    #preparamos el conjunto de datos
    def prepareDataSet(self):

        self.dataResponse = self.dataSet[self.featureClass]
        classArray = list(set(self.dataResponse))#evaluamos si es arreglo binario o no

        self.kindDataSet = 1

        if len(classArray) ==2:
            self.kindDataSet =1
        else:
            self.kindDataSet =2

        #hacemos la transformacion de la clase
        #transformamos la clase si presenta atributos discretos
        transformData = transformDataClass.transformClass(self.dataResponse)
        self.target = transformData.transformData

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

    #metodo que permite aplicar los metodos lineales: SVM, NuSVM
    def applyTraining(self):

        #generamos una lista con los valores obtenidos...
        header = ["Algorithm", "Params", "Validation", "Accuracy", "Recall", "Precision", "F1"]
        AccuracyList = []
        PrecisionList = []
        RecallList = []
        F1List = []

        #NuSVC
        for kernel in ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed']:
            for nu in [0.01, 0.05, 0.1, 0.5]:
                for degree in range(3, 15):
                    try:
                        print "Excec NuSVM ",
                        params = "kernel:%s-nu:%f-degree:%d" % (kernel, nu, degree)
                        print params
                        nuSVM = NuSVM.NuSVM(self.data, self.target, kernel, nu, degree, 0.01, 10)
                        nuSVM.trainingMethod(self.kindDataSet)
                        AccuracyList.append(nuSVM.performanceData.scoreData[3])
                        PrecisionList.append(nuSVM.performanceData.scoreData[4])
                        RecallList.append(nuSVM.performanceData.scoreData[5])
                        F1List.append(nuSVM.performanceData.scoreData[6])
                    except:
                        pass
                    break
                break
            break


        #SVC
        for kernel in ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed']:
            for C_value in [0.01, 0.05, 0.1, 0.5]:
                for degree in range(3, 15):
                    try:
                        print "Excec SVM ",
                        svm = SVM.SVM(self.data, self.target, kernel, C_value, degree, 0.01, 10)
                        params = "kernel:%s-c:%f-degree:%d" % (kernel, C_value, degree)
                        print params
                        svm.trainingMethod(self.kindDataSet)
                        AccuracyList.append(svm.performanceData.scoreData[3])
                        PrecisionList.append(nuSVM.performanceData.scoreData[4])
                        RecallList.append(nuSVM.performanceData.scoreData[5])
                        F1List.append(nuSVM.performanceData.scoreData[6])
                    except:
                        pass
                    break
                break
            break

        try:
            print max(AccuracyList)
            print max(PrecisionList)
            print max(RecallList)
            print max(F1List)
            if max(AccuracyList)<=self.threshold:
                return 0#no lineal
            else:
                return 1#lineal
        except:
            return -1#error
