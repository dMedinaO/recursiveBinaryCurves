########################################################################
#
# Script that allows to execute all the algorithms with different variations
# to generate classifiers and their respective models, stores in a csv
# file the data that is generated with regarding the various measures that
# are obtained as a result of the process
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

import sys
import pandas as pd
import numpy as np
import time
import datetime
import json
import argparse

from modulesNLM.supervised_learning_analysis import AdaBoost
from modulesNLM.supervised_learning_analysis import Baggin
from modulesNLM.supervised_learning_analysis import BernoulliNB
from modulesNLM.supervised_learning_analysis import DecisionTree
from modulesNLM.supervised_learning_analysis import GaussianNB
from modulesNLM.supervised_learning_analysis import Gradient
from modulesNLM.supervised_learning_analysis import knn
from modulesNLM.supervised_learning_analysis import MLP
from modulesNLM.supervised_learning_analysis import NuSVM
from modulesNLM.supervised_learning_analysis import RandomForest
from modulesNLM.supervised_learning_analysis import SVM
from modulesNLM.statistics_analysis import summaryStatistic

#utils para el manejo de set de datos y su normalizacion
from modulesNLM.utils import transformDataClass
from modulesNLM.utils import transformFrequence
from modulesNLM.utils import ScaleNormalScore
from modulesNLM.utils import ScaleMinMax
from modulesNLM.utils import ScaleDataSetLog
from modulesNLM.utils import ScaleLogNormalScore

from modulesNLM.utils import summaryScanProcess
from modulesNLM.utils import responseResults
from modulesNLM.utils import encodingFeatures
from modulesNLM.utils import processParamsDict

from modulesNLM.utils import createDataSetForTraining
from modulesNLM.utils import checkEvalKValue
from modulesNLM.supervised_learning_analysis import evalTraining

#funcion que permite calcular los estadisticos de un atributo en el set de datos, asociados a las medidas de desempeno
def estimatedStatisticPerformance(summaryObject, attribute):

    statistic = summaryObject.calculateValuesForColumn(attribute)
    row = [attribute, statistic['mean'], statistic['std'], statistic['var'], statistic['max'], statistic['min']]

    return row

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataSet", help="full path and name to acces dataSet input process", required=True)
parser.add_argument("-p", "--pathResult", help="full path for save results", required=True)
parser.add_argument("-m", "--performance", help="performance selected model", required=True)
parser.add_argument("-r", "--response", help="name of column with response values in dataset", required=True)
parser.add_argument("-t", "--threshold", type=float, help="threshold of minimus values expected form model generated", required=True)
parser.add_argument("-k", "--kValueData", type=int, help="Value for cross validation, this value most be higher or equal 2", required=True)

args = parser.parse_args()

#hacemos las validaciones asociadas a si existe el directorio y el set de datos
processData = responseResults.responseProcess()#parser y checks...

if (processData.validatePath(args.pathResult) == 0):

    if (processData.validateDataSetExist(args.dataSet) == 0):

        #recibimos los parametros desde la terminal...
        dataSet = pd.read_csv(args.dataSet)

        pathResponse = args.pathResult
        response = args.response
        threshold = args.threshold
        kValueData = args.kValueData

        #valores iniciales
        start_time = time.time()
        inicio = datetime.datetime.now()
        iteracionesCorrectas = 0
        iteracionesIncorrectas = 0

        #procesamos el set de datos para obtener la columna respuesta y la matriz de datos a entrenar
        target = dataSet[response]
        del dataSet[response]

        #transformamos la clase si presenta atributos discretos
        transformData = transformDataClass.transformClass(target)
        target = transformData.transformData
        dictTransform = transformData.dictTransform
        classArray = list(set(target))#evaluamos si es arreglo binario o no

        kindDataSet = 1

        if len(classArray) ==2:
            kindDataSet =1
        else:
            kindDataSet =2

        #ahora transformamos el set de datos por si existen elementos discretos...
        #transformDataSet = transformFrequence.frequenceData(dataValues)
        #dataSetNewFreq = transformDataSet.dataTransform
        encoding = encodingFeatures.encodingFeatures(dataSet, 20)
        encoding.evaluEncoderKind()
        dataSetNewFreq = encoding.dataSet

        #ahora aplicamos el procesamiento segun lo expuesto
        applyNormal = ScaleNormalScore.applyNormalScale(dataSetNewFreq)
        data = applyNormal.dataTransform

        #obtenemos el dataset de entrenamiento y validacion, junto con los arreglos correspondientes de respuestas
        getDataProcess = createDataSetForTraining.createDataSet(data, target)
        dataSetTraining = getDataProcess.dataSetTraining
        classTraining =  getDataProcess.classTraining

        dataSetTesting = getDataProcess.dataSetTesting
        classTesting = getDataProcess.classTesting

        if kValueData == -1:
            validation = "LOU"
        else:
            validation = "CV-"+str(kValueData)

        #generamos una lista con los valores obtenidos...
        header = ["Algorithm", "Params", "Validation", "Accuracy", "Recall", "Precision", "F1"]
        matrixResponse = []

        #AdaBoost
        for algorithm in ['SAMME', 'SAMME.R']:
            for n_estimators in [10,50,100,200,500,1000,1500,2000]:
                #try:
                print "Excec AdaBoost with %s - %d" % (algorithm, n_estimators)
                AdaBoostObject = AdaBoost.AdaBoost(data, target, n_estimators, algorithm, kValueData)
                AdaBoostObject.trainingMethod(kindDataSet)
                params = "Algorithm:%s-n_estimators:%d" % (algorithm, n_estimators)
                row = ["AdaBoostClassifier", params, validation, AdaBoostObject.performanceData.scoreData[3], AdaBoostObject.performanceData.scoreData[4], AdaBoostObject.performanceData.scoreData[5], AdaBoostObject.performanceData.scoreData[6]]
                matrixResponse.append(row)
                iteracionesCorrectas+=1
                #except:
                #    iteracionesIncorrectas+=1
                #    pass

        #Baggin
        for bootstrap in [True, False]:
            for n_estimators in [10,50,100,200,500,1000,1500,2000]:
                try:
                    print "Excec Bagging with %s - %d" % (bootstrap, n_estimators)
                    bagginObject = Baggin.Baggin(data,target,n_estimators, bootstrap,kValueData)
                    bagginObject.trainingMethod(kindDataSet)
                    params = "bootstrap:%s-n_estimators:%d" % (str(bootstrap), n_estimators)
                    row = ["Bagging", params, validation, bagginObject.performanceData.scoreData[3], bagginObject.performanceData.scoreData[4], bagginObject.performanceData.scoreData[5], bagginObject.performanceData.scoreData[6]]
                    matrixResponse.append(row)
                    iteracionesCorrectas+=1
                    print row
                except:
                    iteracionesIncorrectas+=1
                    pass

        #BernoulliNB
        try:
            bernoulliNB = BernoulliNB.Bernoulli(data, target, kValueData)
            bernoulliNB.trainingMethod(kindDataSet)
            print "Excec Bernoulli Default Params"
            params = "Default"
            row = ["BernoulliNB", params, validation, bernoulliNB.performanceData.scoreData[3], bernoulliNB.performanceData.scoreData[4], bernoulliNB.performanceData.scoreData[5], bernoulliNB.performanceData.scoreData[6]]
            matrixResponse.append(row)
            iteracionesCorrectas+=1
            print row
        except:
            iteracionesIncorrectas+=1
            pass

        #DecisionTree
        for criterion in ['gini', 'entropy']:
            for splitter in ['best', 'random']:
                try:
                    print "Excec DecisionTree with %s - %s" % (criterion, splitter)
                    decisionTreeObject = DecisionTree.DecisionTree(data, target, criterion, splitter,kValueData)
                    decisionTreeObject.trainingMethod(kindDataSet)
                    params = "criterion:%s-splitter:%s" % (criterion, splitter)
                    row = ["DecisionTree", params, validation, decisionTreeObject.performanceData.scoreData[3], decisionTreeObject.performanceData.scoreData[4], decisionTreeObject.performanceData.scoreData[5], decisionTreeObject.performanceData.scoreData[6]]
                    matrixResponse.append(row)
                    iteracionesCorrectas+=1
                    print row
                except:
                    iteracionesIncorrectas+=1
                    pass

        try:
            #GaussianNB
            gaussianObject = GaussianNB.Gaussian(data, target, kValueData)
            gaussianObject.trainingMethod(kindDataSet)
            print "Excec GaussianNB Default Params"
            params = "Default"

            row = ["GaussianNB", params, validation, gaussianObject.performanceData.scoreData[3], gaussianObject.performanceData.scoreData[4], gaussianObject.performanceData.scoreData[5], gaussianObject.performanceData.scoreData[6]]
            matrixResponse.append(row)
        except:
            pass

        #gradiente
        for loss in ['deviance', 'exponential']:
            for n_estimators in [10,50,100,200,500,1000,1500,2000]:
                try:
                    print "Excec GradientBoostingClassifier with %s - %d - %d - %d" % (loss, n_estimators, 2, 1)
                    gradientObject = Gradient.Gradient(data,target,n_estimators, loss, min_samples_split, min_samples_leaf, kValueData)
                    gradientObject.trainingMethod(kindDataSet)
                    params = "n_estimators:%d-loss:%s-min_samples_split:%d-min_samples_leaf:%d" % (n_estimators, loss, min_samples_split, min_samples_leaf)
                    row = ["GradientBoostingClassifier", params, validation, gradientObject.performanceData.scoreData[3], gradientObject.performanceData.scoreData[4], gradientObject.performanceData.scoreData[5], gradientObject.performanceData.scoreData[6]]
                    matrixResponse.append(row)
                    iteracionesCorrectas+=1
                    print row
                except:
                    iteracionesIncorrectas+=1
                    pass

        #knn
        for n_neighbors in range(1,11):
            for algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']:
                for metric in ['minkowski', 'euclidean']:
                    for weights in ['uniform', 'distance']:
                        try:
                            print "Excec KNeighborsClassifier with %d - %s - %s - %s" % (n_neighbors, algorithm, metric, weights)
                            knnObect = knn.knn(data, target, n_neighbors, algorithm, metric,  weights,kValueData)
                            knnObect.trainingMethod(kindDataSet)

                            params = "n_neighbors:%d-algorithm:%s-metric:%s-weights:%s" % (n_neighbors, algorithm, metric, weights)
                            row = ["KNeighborsClassifier", params, validation, knnObect.performanceData.scoreData[3], knnObect.performanceData.scoreData[4], knnObect.performanceData.scoreData[5], knnObect.performanceData.scoreData[6]]
                            matrixResponse.append(row)
                            iteracionesCorrectas+=1
                            print row
                        except:
                            iteracionesIncorrectas+=1
                            pass

        #NuSVC
        for kernel in ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed']:
            for nu in [0.01, 0.05, 0.1, 0.5]:
                for degree in range(3, 15):
                    try:
                        print "Excec NuSVM"
                        nuSVM = NuSVM.NuSVM(data, target, kernel, nu, degree, 0.01, kValueData)
                        nuSVM.trainingMethod(kindDataSet)
                        params = "kernel:%s-nu:%f-degree:%d" % (kernel, nu, degree)
                        row = ["NuSVM", params, validation, nuSVM.performanceData.scoreData[3], nuSVM.performanceData.scoreData[4], nuSVM.performanceData.scoreData[5], nuSVM.performanceData.scoreData[6]]
                        matrixResponse.append(row)
                        iteracionesCorrectas+=1
                        print row
                    except:
                        iteracionesIncorrectas+=1
                        pass

        #SVC
        for kernel in ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed']:
            for C_value in [0.01, 0.05, 0.1, 0.5]:
                for degree in range(3, 15):
                    try:
                        print "Excec SVM"
                        svm = SVM.SVM(data, target, kernel, C_value, degree, 0.01, kValueData)
                        svm.trainingMethod(kindDataSet)
                        params = "kernel:%s-c:%f-degree:%d" % (kernel, C_value, degree)
                        row = ["SVM", params, validation, svm.performanceData.scoreData[3], svm.performanceData.scoreData[4], svm.performanceData.scoreData[5], svm.performanceData.scoreData[6]]
                        matrixResponse.append(row)
                        iteracionesCorrectas+=1
                        print row
                    except:
                        iteracionesIncorrectas+=1
                        pass

        #RF
        for n_estimators in [10,50,100,200,500,1000,1500,2000]:
            for criterion in ['gini', 'entropy']:
                for bootstrap in [True, False]:
                    try:
                        print "Excec RF"
                        rf = RandomForest.RandomForest(data, target, n_estimators, criterion, 2, 1, bootstrap, kValueData)
                        rf.trainingMethod(kindDataSet)

                        params = "n_estimators:%d-criterion:%s-min_samples_split:%d-min_samples_leaf:%d-bootstrap:%s" % (n_estimators, criterion, 2, 1, str(bootstrap))
                        row = ["RandomForestClassifier", params, validation, rf.performanceData.scoreData[3], rf.performanceData.scoreData[4], rf.performanceData.scoreData[5], rf.performanceData.scoreData[6]]
                        matrixResponse.append(row)
                        iteracionesCorrectas+=1
                        print row
                    except:
                        iteracionesIncorrectas+=1
                        pass

        try:
            #generamos el export de la matriz convirtiendo a data frame
            dataFrame = pd.DataFrame(matrixResponse, columns=header)

            #generamos el nombre del archivo
            nameFileExport = "%ssummaryProcessJob.csv" % (pathResponse)
            dataFrame.to_csv(nameFileExport, index=False)

            #estimamos los estadisticos resumenes para cada columna en el header
            #instanciamos el object
            statisticObject = summaryStatistic.createStatisticSummary(nameFileExport)
            matrixSummaryStatistic = []

            #"Accuracy", "Recall", "Precision", "Neg_log_loss", "F1", "FBeta"
            matrixSummaryStatistic.append(estimatedStatisticPerformance(statisticObject, 'Accuracy'))
            matrixSummaryStatistic.append(estimatedStatisticPerformance(statisticObject, 'Recall'))
            matrixSummaryStatistic.append(estimatedStatisticPerformance(statisticObject, 'Precision'))
            matrixSummaryStatistic.append(estimatedStatisticPerformance(statisticObject, 'F1'))

            #generamos el nombre del archivo
            dataFrame = pd.DataFrame(matrixSummaryStatistic, columns=['Performance','Mean', 'STD', 'Variance', 'MAX', 'MIN'])
            nameFileExport2 = "%sstatisticPerformance.csv" % (pathResponse)
            dataFrame.to_csv(nameFileExport2, index=False)

            #generamos el proceso estadisitico
            summaryObject = summaryScanProcess.summaryProcessClusteringScan(nameFileExport, pathResponse, ['Accuracy', 'Recall', 'Precision', 'F1'])
            summaryObject.createHistogram()
            summaryObject.createRankingFile()

            finishTime = time.time() - start_time
            termino = datetime.datetime.now()

            dictionary = {}
            dictionary.update({"inicio": str(inicio)})
            dictionary.update({"termino": str(termino)})
            dictionary.update({"ejecucion": finishTime})
            dictionary.update({"iteracionesCorrectas": iteracionesCorrectas})
            dictionary.update({"iteracionesIncorrectas": iteracionesIncorrectas})
            dictionary.update({"performanceSelected": args.performance})

            #agrego la informacion de los mejores modelos para cada medida de desempeno
            processModels = processParamsDict.processParams(pathResponse, ['Accuracy', 'Recall', 'Precision', 'F1'])
            processModels.getBestModels()
            dictionary.update({"modelSelecetd":processModels.listModels})

            nameFileExport = "%ssummaryProcess.json" % (pathResponse)
            with open(nameFileExport, 'w') as fp:
                json.dump(dictionary, fp)
        except:
            print "Error during exec program"
    else:
        print "Data set input not exist, please check the input for name file data set"
else:
    print "Path result not exist, please check input for path result"
