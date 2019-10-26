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

'''
script que permite ejecutar todos los algoritmos con distintas variaciones para generar
clasificadores y sus respectivos modelos, almacena en un archivo csv la data que se genera con
respecto a las diversas medidas que se obtienen como resultado del proceso...
'''

import sys
import pandas as pd
import numpy as np
import time
import datetime
import json
import argparse

from modulesNLM.supervised_learning_predicction import AdaBoost
from modulesNLM.supervised_learning_predicction import Baggin
from modulesNLM.supervised_learning_predicction import DecisionTree
from modulesNLM.supervised_learning_predicction import Gradient
from modulesNLM.supervised_learning_predicction import knn_regression
from modulesNLM.supervised_learning_predicction import MLP
from modulesNLM.supervised_learning_predicction import NuSVR
from modulesNLM.supervised_learning_predicction import RandomForest
from modulesNLM.supervised_learning_predicction import SVR
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

#para evaluar la performance
from modulesNLM.supervised_learning_predicction import performanceData
from modulesNLM.utils import processParamsDict

#funcion que permite calcular los estadisticos de un atributo en el set de datos, asociados a las medidas de desempeno
def estimatedStatisticPerformance(summaryObject, attribute):

    statistic = summaryObject.calculateValuesForColumn(attribute)
    row = [attribute, statistic['mean'], statistic['std'], statistic['var'], statistic['max'], statistic['min']]

    return row

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataSet", help="full path and name to acces dataSet input process", required=True)
parser.add_argument("-p", "--pathResult", help="full path for save results", required=True)
parser.add_argument("-m", "--performance", help="performance selected model", required=True)

args = parser.parse_args()

#hacemos las validaciones asociadas a si existe el directorio y el set de datos
processData = responseResults.responseProcess()#parser y checks...

if (processData.validatePath(args.pathResult) == 0):

    if (processData.validateDataSetExist(args.dataSet) == 0):

        #recibimos los parametros desde la terminal...
        dataSet = pd.read_csv(args.dataSet)
        pathResponse = args.pathResult

        #valores iniciales
        start_time = time.time()
        inicio = datetime.datetime.now()
        iteracionesCorrectas = 0
        iteracionesIncorrectas = 0

        #procesamos el set de datos para obtener los atributos y las clases...
        columnas=dataSet.columns.tolist()
        x=columnas[len(columnas)-1]
        target=dataSet[x]#clases
        y=columnas[0:len(columnas)-1]
        data=dataSet[y]#atributos

        #transformamos la clase si presenta atributos discretos
        transformData = transformDataClass.transformClass(target)
        target = transformData.transformData

        #ahora transformamos el set de datos por si existen elementos discretos...
        #transformDataSet = transformFrequence.frequenceData(data)
        #dataSetNewFreq = transformDataSet.dataTransform
        encoding = encodingFeatures.encodingFeatures(data, 20)
        encoding.evaluEncoderKind()
        dataSetNewFreq = encoding.dataSet
        #ahora aplicamos el procesamiento segun lo expuesto
        applyNormal = ScaleNormalScore.applyNormalScale(dataSetNewFreq)
        data = applyNormal.dataTransform

        #generamos una lista con los valores obtenidos...
        header = ["Algorithm", "Params", "R_Score", "Pearson", "Spearman", "Kendalltau"]
        matrixResponse = []

        #comenzamos con las ejecuciones...

        #AdaBoost
        for loss in ['linear', 'squar', 'exponential']:
            for n_estimators in [10,50,100,200,500,1000,1500,2000]:
                try:
                    print "Excec AdaBoostRegressor with %s - %d" % (loss, n_estimators)
                    AdaBoostObject = AdaBoost.AdaBoost(data, target, n_estimators, loss)
                    AdaBoostObject.trainingMethod()

                    #obtenemos el restante de performance
                    performanceValues = performanceData.performancePrediction(target, AdaBoostObject.predicctions.tolist())
                    pearsonValue = performanceValues.calculatedPearson()['pearsonr']
                    spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
                    kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

                    if pearsonValue == "ERROR":
                        pearsonValue=0
                    if spearmanValue == "ERROR":
                        spearmanValue=0
                    if kendalltauValue == "ERROR":
                        kendalltauValue=0

                    params = "loss:%s-n_estimators:%d" % (loss, n_estimators)
                    row = ["AdaBoostRegressor", params, AdaBoostObject.r_score, pearsonValue, spearmanValue, kendalltauValue]
                    matrixResponse.append(row)
                    iteracionesCorrectas+=1
                except:
                    iteracionesIncorrectas+=1
                    pass
                break
            break

        #Baggin
        for bootstrap in [True, False]:
            for n_estimators in [10,50,100,200,500,1000,1500,2000]:
                try:
                    print "Excec Bagging with %s - %d" % (bootstrap, n_estimators)
                    bagginObject = Baggin.Baggin(data,target,n_estimators, bootstrap)
                    bagginObject.trainingMethod()

                    performanceValues = performanceData.performancePrediction(target, bagginObject.predicctions.tolist())
                    pearsonValue = performanceValues.calculatedPearson()['pearsonr']
                    spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
                    kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

                    if pearsonValue == "ERROR":
                        pearsonValue=0
                    if spearmanValue == "ERROR":
                        spearmanValue=0
                    if kendalltauValue == "ERROR":
                        kendalltauValue=0

                    params = "bootstrap:%s-n_estimators:%d" % (str(bootstrap), n_estimators)
                    row = ["Bagging", params, bagginObject.r_score, pearsonValue, spearmanValue, kendalltauValue]

                    matrixResponse.append(row)
                    iteracionesCorrectas+=1
                except:
                    iteracionesIncorrectas+=1
                    pass
                break
            break


        #DecisionTree
        for criterion in ['mse', 'friedman_mse', 'mae']:
            for splitter in ['best', 'random']:
                try:
                    print "Excec DecisionTree with %s - %s" % (criterion, splitter)
                    decisionTreeObject = DecisionTree.DecisionTree(data, target, criterion, splitter)
                    decisionTreeObject.trainingMethod()

                    performanceValues = performanceData.performancePrediction(target, decisionObject.predicctions.tolist())
                    pearsonValue = performanceValues.calculatedPearson()['pearsonr']
                    spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
                    kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

                    if pearsonValue == "ERROR":
                        pearsonValue=0
                    if spearmanValue == "ERROR":
                        spearmanValue=0
                    if kendalltauValue == "ERROR":
                        kendalltauValue=0

                    params = "criterion:%s-splitter:%s" % (criterion, splitter)
                    row = ["DecisionTree", params, decisionTreeObject.r_score, pearsonValue, spearmanValue, kendalltauValue]
                    matrixResponse.append(row)
                    iteracionesCorrectas+=1
                except:
                    iteracionesIncorrectas+=1
                    pass
                break
            break

        #gradiente
        for loss in ['ls', 'lad', 'huber', 'quantile']:
            for criterion in ['friedman_mse', 'mse', 'mae']:
                for n_estimators in [10,50,100,200,500,1000,1500,2000]:
                    for min_samples_split in range (2, 11):
                        for min_samples_leaf in range(1, 11):
                            try:
                                print "Excec GradientBoostingRegressor with %s - %d - %d - %d" % (loss, n_estimators, min_samples_split, min_samples_leaf)
                                gradientObject = Gradient.Gradient(data,target,n_estimators, loss, criterion, min_samples_split, min_samples_leaf)
                                gradientObject.trainingMethod()

                                performanceValues = performanceData.performancePrediction(target, gradientObject.predicctions.tolist())
                                pearsonValue = performanceValues.calculatedPearson()['pearsonr']
                                spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
                                kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

                                if pearsonValue == "ERROR":
                                    pearsonValue=0
                                if spearmanValue == "ERROR":
                                    spearmanValue=0
                                if kendalltauValue == "ERROR":
                                    kendalltauValue=0

                                params = "criterion:%s-n_estimators:%d-loss:%s-min_samples_split:%d-min_samples_leaf:%d" % (criterion, n_estimators, loss, min_samples_split, min_samples_leaf)
                                row = ["GradientBoostingClassifier", params, gradientObject.r_score, pearsonValue, spearmanValue, kendalltauValue]
                                matrixResponse.append(row)
                                iteracionesCorrectas+=1
                            except:
                                iteracionesIncorrectas+=1
                                pass
                            break
                        break
                    break
                break
            break

        #knn
        for n_neighbors in range(1,11):
            for algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']:
                for metric in ['minkowski', 'euclidean']:
                    for weights in ['uniform', 'distance']:
                        try:
                            print "Excec KNeighborsRegressor with %d - %s - %s - %s" % (n_neighbors, algorithm, metric, weights)
                            knnObect = knn_regression.KNN_Model(data, target, n_neighbors, algorithm, metric,  weights)
                            knnObect.trainingMethod()

                            performanceValues = performanceData.performancePrediction(target, knnObect.predicctions.tolist())
                            pearsonValue = performanceValues.calculatedPearson()['pearsonr']
                            spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
                            kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

                            if pearsonValue == "ERROR":
                                pearsonValue=0
                            if spearmanValue == "ERROR":
                                spearmanValue=0
                            if kendalltauValue == "ERROR":
                                kendalltauValue=0

                            params = "n_neighbors:%d-algorithm:%s-metric:%s-weights:%s" % (n_neighbors, algorithm, metric, weights)
                            row = ["KNeighborsRegressor", params, knnObect.r_score, pearsonValue, spearmanValue, kendalltauValue]
                            matrixResponse.append(row)
                            iteracionesCorrectas+=1
                        except:
                            iteracionesIncorrectas+=1
                            pass
                        break
                    break
                break
            break

        '''
        #MLP
        #activation, solver, learning_rate, hidden_layer_sizes_a,hidden_layer_sizes_b,hidden_layer_sizes_c, alpha, max_iter, shuffle
        for activation in ['identity', 'logistic', 'tanh', 'relu']:
            for solver in ['lbfgs', 'sgd', 'adam']:
                for learning_rate in ['constant', 'invscaling', 'adaptive']:
                    for hidden_layer_sizes_a in  range(1,11):
                        for hidden_layer_sizes_b in range(1,11):
                            for hidden_layer_sizes_c in range(1, 11):
                                for alpha in [0.001, 0.002, 0.01, 0.02, 0.1, 0.2]:
                                    for max_iter in [100,200,500,1000,1500]:
                                        for shuffle in [True, False]:
                                            try:
                                                print "Excec MLP"
                                                MLPObject = MLP.MLP(data, target, activation, solver, learning_rate, hidden_layer_sizes_a,hidden_layer_sizes_b,hidden_layer_sizes_c, alpha, max_iter, shuffle)
                                                MLPObject.trainingMethod()

                                                performanceValues = performanceData.performancePrediction(target, MLPObject.predicctions.tolist())
                                                pearsonValue = performanceValues.calculatedPearson()['pearsonr']
                                                spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
                                                kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

                                                params = "activation:%s-solver:%s-learning:%s-hidden_layer_sizes:%d+%d+%d-alpha:%f-max_iter:%d-shuffle:%s" % (activation, solver, learning_rate, hidden_layer_sizes_a, hidden_layer_sizes_b, hidden_layer_sizes_c, alpha, max_iter, str(shuffle))
                                                row = ["MLPRegressor", params, MLPObject.r_score, pearsonValue, spearmanValue, kendalltauValue]
                                                matrixResponse.append(row)
                                                iteracionesCorrectas+=1
                                            except:
                                                iteracionesIncorrectas+=1
                                                pass
                                            print matrixResponse
        '''
        #NuSVR
        for kernel in ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed']:
            for nu in [0.01, 0.05, 0.1, 0.5]:
                for degree in range(3, 15):
                    for gamma in [0.01, 0.1, 1, 10, 100]:
                        try:
                            print "Excec NuSVM"
                            nuSVM = NuSVR.NuSVRModel(data, target, kernel, degree, gamma, nu)
                            nuSVM.trainingMethod()

                            performanceValues = performanceData.performancePrediction(target, nuSVM.predicctions.tolist())
                            pearsonValue = performanceValues.calculatedPearson()['pearsonr']
                            spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
                            kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

                            if pearsonValue == "ERROR":
                                pearsonValue=0
                            if spearmanValue == "ERROR":
                                spearmanValue=0
                            if kendalltauValue == "ERROR":
                                kendalltauValue=0

                            params = "kernel:%s-nu:%f-degree:%d-gamma:%f" % (kernel, nu, degree, gamma)
                            row = ["NuSVM", params, nuSVM.r_score, pearsonValue, spearmanValue, kendalltauValue]
                            matrixResponse.append(row)
                            iteracionesCorrectas+=1
                        except:
                            iteracionesIncorrectas+=1
                            pass
                        break
                    break
                break
            break

        #SVC
        for kernel in ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed']:
            for degree in range(3, 15):
                for gamma in [0.01, 0.1, 1, 10, 100]:
                    try:
                        print "Excec SVM"
                        svm = SVR.SVRModel(data, target, kernel, degree, gamma)
                        svm.trainingMethod()

                        performanceValues = performanceData.performancePrediction(target, svm.predicctions.tolist())
                        pearsonValue = performanceValues.calculatedPearson()['pearsonr']
                        spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
                        kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

                        if pearsonValue == "ERROR":
                            pearsonValue=0
                        if spearmanValue == "ERROR":
                            spearmanValue=0
                        if kendalltauValue == "ERROR":
                            kendalltauValue=0

                        params = "kernel:%s-degree:%d-gamma:%f" % (kernel, degree, gamma)
                        row = ["SVM", params, svm.r_score, pearsonValue, spearmanValue, kendalltauValue]
                        matrixResponse.append(row)
                        iteracionesCorrectas+=1
                    except:
                        iteracionesIncorrectas+=1
                        pass
                    break
                break
            break

        #RF
        for n_estimators in [10,50,100,200,500,1000,1500,2000]:
            for criterion in ['mse', 'mae']:
                for min_samples_split in range (2, 11):
                    for min_samples_leaf in range(1, 11):
                        for bootstrap in [True, False]:
                            try:
                                print "Excec RF"
                                rf = RandomForest.RandomForest(data, target, n_estimators, criterion, min_samples_split, min_samples_leaf, bootstrap)
                                rf.trainingMethod()

                                performanceValues = performanceData.performancePrediction(target, rf.predicctions.tolist())
                                pearsonValue = performanceValues.calculatedPearson()['pearsonr']
                                spearmanValue = performanceValues.calculatedSpearman()['spearmanr']
                                kendalltauValue = performanceValues.calculatekendalltau()['kendalltau']

                                if pearsonValue == "ERROR":
                                    pearsonValue=0
                                if spearmanValue == "ERROR":
                                    spearmanValue=0
                                if kendalltauValue == "ERROR":
                                    kendalltauValue=0

                                params = "n_estimators:%d-criterion:%s-min_samples_split:%d-min_samples_leaf:%d-bootstrap:%s" % (n_estimators, criterion, min_samples_split, min_samples_leaf, str(bootstrap))
                                row = ["RandomForestRegressor", params, rf.r_score, pearsonValue, spearmanValue, kendalltauValue]
                                matrixResponse.append(row)
                                iteracionesCorrectas+=1
                            except:
                                iteracionesIncorrectas+=1
                                pass
                            break
                        break
                    break
                break
            break

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
        matrixSummaryStatistic.append(estimatedStatisticPerformance(statisticObject, 'R_Score'))
        matrixSummaryStatistic.append(estimatedStatisticPerformance(statisticObject, 'Pearson'))
        matrixSummaryStatistic.append(estimatedStatisticPerformance(statisticObject, 'Spearman'))
        matrixSummaryStatistic.append(estimatedStatisticPerformance(statisticObject, 'Kendalltau'))

        #generamos el nombre del archivo
        dataFrame = pd.DataFrame(matrixSummaryStatistic, columns=['Performance','Mean', 'STD', 'Variance', 'MAX', 'MIN'])
        nameFileExport2 = "%sstatisticPerformance.csv" % (pathResponse)
        dataFrame.to_csv(nameFileExport2, index=False)

        summaryObject = summaryScanProcess.summaryProcessClusteringScan(nameFileExport, pathResponse, ['R_Score', 'Pearson', 'Spearman', 'Kendalltau'])
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
        processModels = processParamsDict.processParams(pathResponse, ['R_Score', 'Pearson', 'Spearman', 'Kendalltau'])
        processModels.getBestModels()
        dictionary.update({"modelSelecetd":processModels.listModels})

        nameFileExport = "%ssummaryProcess.json" % (pathResponse)
        with open(nameFileExport, 'w') as fp:
            json.dump(dictionary, fp)
    else:
        print "Data set input not exist, please check the input for name file data set"
else:
    print "Path result not exist, please check input for path result"
