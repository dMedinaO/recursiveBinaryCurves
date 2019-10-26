import random
import pandas as pd
import json
from modulesNLM.clustering_analysis import evaluationClustering

class checkRandomsPartitions(object):

    def __init__(self, numberPartitions, pathOutput, dataSet, performanceC, performanceS):

        self.numberPartitions = numberPartitions
        self.pathOutput = pathOutput
        self.dataSet = dataSet
        self.performanceC = performanceC#calinski obtenido a partir de las divisiones del metodo propuesto
        self.performanceS = performanceS#siluetas obtenido a partir de las divisiones del metodo propuesto

    #funcion que permite crear los tamanos de los conjuntos aleatorios
    def createSizePartitionsRandom(self, numberGroup):

        maxData = int(len(self.dataSet)/float(numberGroup))#la muestra dividida en el total de grupos
        minData = int(len(self.dataSet)*0.1)#el 5% de la muestra

        splitter = []
        for j in range(1, numberGroup):
            splitter.append(random.randrange(minData, maxData))
        residue = len(self.dataSet)-sum(splitter)
        splitter.append(residue)

        return splitter

    #funcion que permite crear la particion aleatoria
    def createPartition(self, randomIndex, splitterPoints):

        matrixResponse = []

        for i in range(len(self.dataSet)):
            row = []
            for key in self.dataSet:
                row.append(self.dataSet[key][randomIndex[i]])
            matrixResponse.append(row)

        #generamos el arreglo de labels
        labels=[]
        indexLabel =0

        for splitter in splitterPoints:
            for i in range(splitter):
                labels.append(indexLabel)
            indexLabel+=1
        return matrixResponse, labels

    #funcion que permite crear las particiones aleatorias y evaluarlas
    def createRandomPartitions(self, numberGroup):

        maxData = int(len(self.dataSet)/float(numberGroup))#la muestra dividida en el total de grupos
        minData = int(len(self.dataSet)*0.05)#el 5% de la muestra

        #generamos el arreglo de indices
        indexArray = []
        for i in range(len(self.dataSet)):
            indexArray.append(i)

        calinskyIndex = []
        siluetasIndex = []

        for i in range(100):#se hace un muestreo de tamano 100

            print "random sample: ", i
            splitter = self.createSizePartitionsRandom(numberGroup)
            random.shuffle(indexArray)#random a los index
            matrixData, labels = self.createPartition(indexArray, splitter)

            #hacemos la evaluacion
            resultEvaluation = evaluationClustering.evaluationClustering(matrixData, labels)#evaluamos...
            calinskyIndex.append(resultEvaluation.calinski)
            siluetasIndex.append(resultEvaluation.siluetas)

        return calinskyIndex, siluetasIndex
    #funcion que genera el proceso completo de las particiones
    def checkRandomElement(self):

        dataFrame = pd.DataFrame()
        dataFrameR = pd.DataFrame()

        for i in range(2, self.numberPartitions):
            print "random splitter: ", i
            calinskyIndex, siluetasIndex = self.createRandomPartitions(i)
            keyC = "calinski_partition_"+str(i)
            keyS = "siluetas_partition_"+str(i)
            dataFrame[keyC] = calinskyIndex
            dataFrame[keyS] = siluetasIndex

            maxCal = max(calinskyIndex)
            maxSil = max(siluetasIndex)

            if maxCal > self.performanceC:
                dataFrameR[keyC] = [0]#indica que no se cumple la hipotesis
            else:
                dataFrameR[keyC] = [1]#indica que se cumple la hipotesis

            if maxSil > self.performanceS:
                dataFrameR[keyS] = [0]#indica que no se cumple la hipotesis
            else:
                dataFrameR[keyS] = [1]#indica que se cumple la hipotesis

        dataFrame.to_csv(self.pathOutput+"performanceRandomSplitter.csv", index=False)
        dataFrameR.to_csv(self.pathOutput+"hipotesisTestCheck.csv", index=False)
