import random
import pandas as pd
import json
from modulesNLM.clustering_analysis import evaluationClustering

class checkMembersInPartitions(object):

    def __init__(self, listSplitter, pathOutput, dataSet, performanceC, performanceS):

        self.listSplitter = listSplitter
        self.pathOutput = pathOutput
        self.dataSet = dataSet
        self.performanceC = performanceC
        self.performanceS = performanceS

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

    #funcion que permite generar las particiones aleatorias y evaluarlas
    def generateRandomPartitions(self):

        #generamos el arreglo de indices
        indexArray = []
        for i in range(len(self.dataSet)):
            indexArray.append(i)

        calinskyIndex = []
        siluetasIndex = []

        for i in range(100):#se hace un muestreo de tamano 100

            print "random sample: ", i

            random.shuffle(indexArray)#random a los index
            matrixData, labels = self.createPartition(indexArray, self.listSplitter)

            #hacemos la evaluacion
            resultEvaluation = evaluationClustering.evaluationClustering(matrixData, labels)#evaluamos...
            calinskyIndex.append(resultEvaluation.calinski)
            siluetasIndex.append(resultEvaluation.siluetas)

        maxCal = max(calinskyIndex)
        maxSil = max(siluetasIndex)

        print maxCal
        print maxSil

        dictResponse = {}

        if maxCal > self.performanceC:
            dictResponse.update({"responseCalinski":0})#no cumple
        else:
            dictResponse.update({"responseCalinski":1})#si cumple

        if maxSil > self.performanceS:
            dictResponse.update({"responseSiluetas":0})#no cumple
        else:
            dictResponse.update({"responseSiluetas":1})#si cumple

        print dictResponse

        nameDoc= self.pathOutput+"response_checkMembers.json"
        with open(nameDoc, 'w') as docExport:
            json.dump(dictResponse, docExport)
