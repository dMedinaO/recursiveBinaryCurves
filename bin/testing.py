import pandas as pd
import sys

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

dataSet = pd.read_csv(sys.argv[1])
dataTesting = pd.read_csv(sys.argv[2])
response = "class"

#hacemos la preparacion y entrenamos segun la data de interes P1 de Thoraric
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

#para el testeo
target2 = dataTesting[response]
del dataTesting[response]

#transformamos la clase si presenta atributos discretos
transformData = transformDataClass.transformClass(target2)
target2 = transformData.transformData
dictTransform2 = transformData.dictTransform
classArray = list(set(target2))#evaluamos si es arreglo binario o no

#ahora transformamos el set de datos por si existen elementos discretos...
#transformDataSet = transformFrequence.frequenceData(dataValues)
#dataSetNewFreq = transformDataSet.dataTransform
encoding = encodingFeatures.encodingFeatures(dataTesting, 20)
encoding.evaluEncoderKind()
dataSetNewFreq2 = encoding.dataSet

#ahora aplicamos el procesamiento segun lo expuesto
applyNormal = ScaleNormalScore.applyNormalScale(dataSetNewFreq2)
dataTesting = applyNormal.dataTransform

#entrenamos los modelos segun corresponda
#Accuracy: KNeighborsClassifier	n_neighbors:6-algorithm:kd_tree-metric:euclidean-weights:uniform
knnObect = knn.knn(data, target, 6, 'kd_tree', 'euclidean',  'uniform',10)
knnObect.trainingMethod(kindDataSet)

listPredictAccuracy = knnObect.knnAlgorithm.predict(dataTesting)


#Recall: KNeighborsClassifier	n_neighbors:6-algorithm:ball_tree-metric:euclidean-weights:uniform
knnObect2 = knn.knn(data, target, 6, 'ball_tree', 'euclidean',  'uniform',10)
knnObect2.trainingMethod(kindDataSet)
listPredictRecall = knnObect2.knnAlgorithm.predict(dataTesting)

#Precision: NuSVM	kernel:rbf-nu:0.050000-degree:14
nuSVM = NuSVM.NuSVM(data, target, 'rbf', 0.05, 14, 0.01, 10)
nuSVM.trainingMethod(kindDataSet)
listPredictPrecision = nuSVM.NuSVMAlgorithm.predict(dataTesting)

#F1: AdaBoostClassifier	Algorithm:SAMME-n_estimators:1500
AdaBoostObject = AdaBoost.AdaBoost(data, target, 1500, 'SAMME', 10)
AdaBoostObject.trainingMethod(kindDataSet)
listPredictF1 = AdaBoostObject.AdaBoostAlgorithm.predict(dataTesting)

matrixResponse = []

for i in range(len(listPredictF1)):
    row = [listPredictF1[i], listPredictRecall[i], listPredictAccuracy[i], listPredictPrecision[i], target2[i]]
    matrixResponse.append(row)

header = ['F1', 'R', 'Acc', 'Pre', 'Real']
dataResponse = pd.DataFrame(matrixResponse, columns=header)

dataResponse.to_csv("responseP1.csv", index=False)
