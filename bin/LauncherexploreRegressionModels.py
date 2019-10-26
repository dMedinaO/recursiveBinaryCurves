########################################################################
# LauncherexploreRegressionModels.py,
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

#para preparar el conjunto de datos
from modulesNLM.utils import createDataSetForTraining

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

        #valores iniciales
        start_time = time.time()
        inicio = datetime.datetime.now()
        iteracionesCorrectas = 0
        iteracionesIncorrectas = 0

        #procesamos el set de datos para obtener la columna respuesta y la matriz de datos a entrenar
        target = dataSet[response]
        del dataSet[response]

        #procesamos la data de interes, asociada a la codificacion de las variables categoricas y la normalizacion del conjunto de datos
        #transformamos la clase si presenta atributos discretos
        transformData = transformDataClass.transformClass(target)
        target = transformData.transformData

        #ahora transformamos el set de datos por si existen elementos categoricos...
        #transformDataSet = transformFrequence.frequenceData(data)
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

        #generamos una lista con los valores obtenidos...
        header = ["Algorithm", "Params", "R_Score", "Pearson", "Spearman", "Kendalltau"]
        matrixResponse = []

        #comenzamos con las ejecuciones...

        #AdaBoost
        for loss in ['linear', 'squar', 'exponential']:
            for n_estimators in [10,50,100,200,500,1000,1500,2000]:
                try:
                    print "Excec AdaBoostRegressor with %s - %d" % (loss, n_estimators)
                    AdaBoostObject = AdaBoost.AdaBoost(dataSetTraining, classTraining, n_estimators, loss)
                    AdaBoostObject.trainingMethod()

                    #obtenemos el restante de performance
                    #usamos el modelo para predecir los elementos y comparamos con respecto al valor del testing
                    predictedValues = AdaBoostObject.AdaBoostModel.predict(dataSetTesting).tolist()
                    rscore = AdaBoostObject.AdaBoostModel.score(dataSetTesting, classTesting)

                    performanceValues = performanceData.performancePrediction(classTesting, predictedValues)
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
                    row = ["AdaBoostRegressor", params, rscore, pearsonValue, spearmanValue, kendalltauValue]
                    matrixResponse.append(row)
                    iteracionesCorrectas+=1
                    print row
                except:
                    iteracionesIncorrectas+=1
                    pass


        #Baggin
        for bootstrap in [True, False]:
            for n_estimators in [10,50,100,200,500,1000,1500,2000]:
                try:
                    print "Excec Bagging with %s - %d" % (bootstrap, n_estimators)
                    bagginObject = Baggin.Baggin(dataSetTraining, classTraining, n_estimators, bootstrap)
                    bagginObject.trainingMethod()

                    predictedValues = bagginObject.bagginModel.predict(dataSetTesting).tolist()
                    rscore = bagginObject.bagginModel.score(dataSetTesting, classTesting)

                    performanceValues = performanceData.performancePrediction(classTesting, predictedValues)
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
                    row = ["Bagging", params, rscore, pearsonValue, spearmanValue, kendalltauValue]

                    matrixResponse.append(row)
                    iteracionesCorrectas+=1
                    print row
                except:
                    iteracionesIncorrectas+=1
                    pass

        #DecisionTree
        for criterion in ['mse', 'friedman_mse', 'mae']:
            for splitter in ['best', 'random']:
                try:
                    print "Excec DecisionTree with %s - %s" % (criterion, splitter)
                    decisionTreeObject = DecisionTree.DecisionTree(dataSetTraining, classTraining, criterion, splitter)
                    decisionTreeObject.trainingMethod()

                    predictedValues = decisionTreeObject.DecisionTreeAlgorithm.predict(dataSetTesting).tolist()
                    rscore = decisionTreeObject.DecisionTreeAlgorithm.score(dataSetTesting, classTesting)

                    performanceValues = performanceData.performancePrediction(classTesting, predictedValues)
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
                    row = ["DecisionTree", params, rscore, pearsonValue, spearmanValue, kendalltauValue]
                    matrixResponse.append(row)
                    iteracionesCorrectas+=1
                    print row
                except:
                    iteracionesIncorrectas+=1
                    pass


        #gradiente
        for loss in ['ls', 'lad', 'huber', 'quantile']:
            for criterion in ['friedman_mse', 'mse', 'mae']:
                for n_estimators in [10,50,100,200,500,1000,1500,2000]:
                    for min_samples_split in range (2, 11):
                        for min_samples_leaf in range(1, 11):
                            try:
                                print "Excec GradientBoostingRegressor with %s - %d - %d - %d" % (loss, n_estimators, min_samples_split, min_samples_leaf)
                                gradientObject = Gradient.Gradient(dataSetTraining,classTraining,n_estimators, loss, criterion, min_samples_split, min_samples_leaf)
                                gradientObject.trainingMethod()

                                predictedValues = gradientObject.GradientAlgorithm.predict(dataSetTesting).tolist()
                                rscore = gradientObject.GradientAlgorithm.score(dataSetTesting, classTesting)

                                performanceValues = performanceData.performancePrediction(classTesting, predictedValues)
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
                                row = ["GradientBoostingClassifier", params, rscore, pearsonValue, spearmanValue, kendalltauValue]
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
                            print "Excec KNeighborsRegressor with %d - %s - %s - %s" % (n_neighbors, algorithm, metric, weights)
                            knnObect = knn_regression.KNN_Model(dataSetTraining, classTraining, n_neighbors, algorithm, metric,  weights)
                            knnObect.trainingMethod()

                            predictedValues = knnObect.KNN_model.predict(dataSetTesting).tolist()
                            rscore = knnObect.KNN_model.score(dataSetTesting, classTesting)

                            performanceValues = performanceData.performancePrediction(classTesting, predictedValues)
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
                            row = ["KNeighborsRegressor", params, rscore, pearsonValue, spearmanValue, kendalltauValue]
                            matrixResponse.append(row)
                            iteracionesCorrectas+=1
                            print row
                        except:
                            iteracionesIncorrectas+=1
                            pass


        #NuSVR
        for kernel in ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed']:
            for nu in [0.01, 0.05, 0.1, 0.5]:
                for degree in range(3, 15):
                    for gamma in [0.01, 0.1, 1, 10, 100]:
                        try:
                            print "Excec NuSVM"
                            nuSVM = NuSVR.NuSVRModel(dataSetTraining, classTraining, kernel, degree, gamma, nu)
                            nuSVM.trainingMethod()

                            predictedValues = nuSVM.SVRAlgorithm.predict(dataSetTesting).tolist()
                            rscore = nuSVM.SVRAlgorithm.score(dataSetTesting, classTesting)

                            performanceValues = performanceData.performancePrediction(classTesting, predictedValues)
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
                            row = ["NuSVM", params, rscore, pearsonValue, spearmanValue, kendalltauValue]
                            matrixResponse.append(row)
                            iteracionesCorrectas+=1
                            print row
                        except:
                            iteracionesIncorrectas+=1
                            pass

        #SVC
        for kernel in ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed']:
            for degree in range(3, 15):
                for gamma in [0.01, 0.1, 1, 10, 100]:
                    try:
                        print "Excec SVM"
                        svm = SVR.SVRModel(dataSetTraining, classTraining, kernel, degree, gamma)
                        svm.trainingMethod()

                        predictedValues = svm.SVRAlgorithm.predict(dataSetTesting).tolist()
                        rscore = svm.SVRAlgorithm.score(dataSetTesting, classTesting)

                        performanceValues = performanceData.performancePrediction(classTesting, predictedValues)
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
                        row = ["SVM", params, rscore, pearsonValue, spearmanValue, kendalltauValue]
                        matrixResponse.append(row)
                        iteracionesCorrectas+=1
                        print row
                    except:
                        iteracionesIncorrectas+=1
                        pass

        for n_estimators in [10,50,100,200,500,1000,1500,2000]:
            for criterion in ['mse', 'mae']:
                for min_samples_split in range (2, 11):
                    for min_samples_leaf in range(1, 11):
                        for bootstrap in [True, False]:
                            try:
                                print "Excec RF"
                                rf = RandomForest.RandomForest(dataSetTraining, classTraining, n_estimators, criterion, min_samples_split, min_samples_leaf, bootstrap)
                                rf.trainingMethod()

                                predictedValues = rf.randomForesModel.predict(dataSetTesting).tolist()
                                rscore = rf.randomForesModel.score(dataSetTesting, classTesting)

                                performanceValues = performanceData.performancePrediction(classTesting, predictedValues)
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
                                row = ["RandomForestRegressor", params, rscore, pearsonValue, spearmanValue, kendalltauValue]
                                matrixResponse.append(row)
                                iteracionesCorrectas+=1
                                print row
                            except:
                                iteracionesIncorrectas+=1
                                pass
                            
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
