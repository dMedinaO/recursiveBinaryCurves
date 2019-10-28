import random
import pandas as pd

class createDataSet(object):

    def __init__(self, dataSet, classResponse):

        self.dataSet = dataSet
        self.classResponse = classResponse
        self.createProportionsData()
        self.createArrayRandomIndex()

        self.createTrainingAndTestingDataSet()

    #metodo que recibe un conjunto de datos, un array de indices el inicio y final y genera una matriz con los datos de un dataFrame
    def createDataFromDataset(self, init, stop):

        matrixData = []

        for i in range(init, stop):

            row = []
            for element in self.dataSet:
                row.append(self.dataSet[element][self.indexArray[i]])
            matrixData.append(row)

        header = []
        for key in self.dataSet:
            header.append(key)

        dataSetResponse = pd.DataFrame(matrixData, columns=header)
        return dataSetResponse

    #metodo que permite crear la variable de las respuestas
    def createDataResponseFromDataSet(self, init, stop):

        arrayResponse = []

        for i in range(init, stop):
            arrayResponse.append(self.classResponse[self.indexArray[i]])

        return arrayResponse

    #metodo que permite determinar la proporcion de ejemplos para el conjunto de entrenamiento y de validacion
    def createProportionsData(self):

        lenData = len(self.dataSet)

        self.valueTraining = int(lenData*0.8)
        self.valueTesting = lenData-self.valueTraining

    #metodo que permite generar indices aleatorios
    def createArrayRandomIndex(self):

        self.indexArray = []

        for i in range(len(self.dataSet)):
            self.indexArray.append(i)

        random.shuffle(self.indexArray)

    #metodo que permite crear los conjuntos de entrenamiento y testeo
    def createTrainingAndTestingDataSet(self):

        self.dataSetTraining = self.createDataFromDataset(0, self.valueTraining)
        self.classTraining = self.createDataResponseFromDataSet(0, self.valueTraining)

        self.dataSetTesting = self.createDataFromDataset(self.valueTraining, len(self.dataSet))
        self.classTesting = self.createDataResponseFromDataSet(self.valueTraining, len(self.dataSet))

    #metodo que permite determinar el valor de la validacion cruzada a utilizar con respecto al tamano del dataset de entrenamiento
    def checkKValueForCrossValidation(self):

        self.kValue = 0

        if self.valueTraining>=150:
            self.kValue=10
        elif self.valueTraining<150 and self.valueTraining>=100:
            self.kValue=5
        elif self.valueTraining <100 and self.valueTraining>=50:
            self.kValue=3
        else:
            self.kValue=-1#this values is associated with a leave one out validation
