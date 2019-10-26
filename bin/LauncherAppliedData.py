import pandas as pd
import sys
from modulesNLM.supervised_learning_analysis import DecisionTree
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

#recibimos el conjunto de datos, formamos el conjunto de datos y hacemos la division entre validacion y testeo
#hacemos el entrenamiento de las particiones y formamos el modelo para asignar ejemplos a la particion y luego formo los
#conjuntos de evaluacion a cada particion

dataPartitions = pd.read_csv(sys.argv[1])
dataSetFull = pd.read_csv(sys.argv[2])
responseOK = sys.argv[3]
pathResult = sys.argv[4]

response = "partitions"

#hacemos el entrenamiento con arboles de decision
#procesamos el set de datos para obtener la columna respuesta y la matriz de datos a entrenar
target = dataPartitions[response]
del dataPartitions[response]
del dataPartitions[responseOK]#representa a la clase de prediccion

#transformamos la clase si presenta atributos discretos
transformData = transformDataClass.transformClass(target)
target = transformData.transformData
dictTransform = transformData.dictTransform
classArray = list(set(target))#evaluamos si es arreglo binario o no

print dictTransform
kindDataSet = 1

if len(classArray) ==2:
    kindDataSet =1
else:
    kindDataSet =2

#ahora transformamos el set de datos por si existen elementos discretos...
transformDataSet = transformFrequence.frequenceData(dataPartitions)
dataSetNewFreq = transformDataSet.dataTransform
#encoding = encodingFeatures.encodingFeatures(dataPartitions, 20)
#encoding.evaluEncoderKind()
#dataSetNewFreq = encoding.dataSet

#ahora aplicamos el procesamiento segun lo expuesto
applyNormal = ScaleNormalScore.applyNormalScale(dataSetNewFreq)
data = applyNormal.dataTransform

#hago el entrenamiento con arboles de decision
decisionTreeObject = DecisionTree.DecisionTree(data, target, 'gini', 'best',10)
decisionTreeObject.trainingMethod(kindDataSet)

print decisionTreeObject.performanceData.scoreData[3], decisionTreeObject.performanceData.scoreData[4], decisionTreeObject.performanceData.scoreData[5], decisionTreeObject.performanceData.scoreData[6]

#del conjunto de datos original, tomamos los elementos y formamos un conjunto de testeo...
#procesamos el set de datos para obtener la columna respuesta y la matriz de datos a entrenar
target = dataSetFull[responseOK]
target2 = dataSetFull[responseOK]
del dataSetFull[responseOK]

#transformamos la clase si presenta atributos discretos
transformData = transformDataClass.transformClass(target)
target = transformData.transformData
dictTransform2 = transformData.dictTransform
classArray = list(set(target))#evaluamos si es arreglo binario o no

kindDataSet = 1

if len(classArray) ==2:
    kindDataSet =1
else:
    kindDataSet =2

#ahora transformamos el set de datos por si existen elementos discretos...
transformDataSet = transformFrequence.frequenceData(dataSetFull)
dataSetNewFreq = transformDataSet.dataTransform
#encoding = encodingFeatures.encodingFeatures(dataSetFull, 20)
#encoding.evaluEncoderKind()
#dataSetNewFreq = encoding.dataSet

#ahora aplicamos el procesamiento segun lo expuesto
applyNormal = ScaleNormalScore.applyNormalScale(dataSetNewFreq)
data = applyNormal.dataTransform

#obtenemos el dataset de entrenamiento y validacion, junto con los arreglos correspondientes de respuestas
getDataProcess = createDataSetForTraining.createDataSet(data, target)
dataSetTraining = getDataProcess.dataSetTraining
classTraining =  getDataProcess.classTraining

dataSetTesting = getDataProcess.dataSetTesting
classTesting = getDataProcess.classTesting

#asignamos los elementos a una particion
responsePredict = decisionTreeObject.DecisionTreeAlgorithm.predict(dataSetTraining)

matrixData = []
header = []
for i in range (getDataProcess.valueTraining):
    row = []
    for key in dataSetFull:
        row.append(dataSetFull[key][getDataProcess.indexArray[i]])
        if i == 0:
            header.append(key)

    #adicionamos la clase y la particion
    row.append(target2[getDataProcess.indexArray[i]])
    row.append(responsePredict[i])
    matrixData.append(row)

header.append(responseOK)

#ahora hacemos la division de los conjuntos de datos
responseUnique = list(set(responsePredict))

for element in responseUnique:

    matrixDataClass=[]

    for row in matrixData:
        if row[-1] == element:
            newRow = []
            for value in range(len(row)-1):
                newRow.append(row[value])
            matrixDataClass.append(newRow)

    #generamos el dataFrame...
    dataFrame = pd.DataFrame(matrixDataClass, columns=header)
    dataFrame.to_csv(pathResult+"partition_"+str(element)+".csv", index=False)
