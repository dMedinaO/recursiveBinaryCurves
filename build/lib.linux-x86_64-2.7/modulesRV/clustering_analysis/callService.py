########################################################################
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

from modulesRV.clustering_analysis import processClustering
from modulesRV.clustering_analysis import evaluationClustering
from modulesRV.clustering_analysis import summaryScan
from modulesRV.checks_module import checkProcessCluster

import pandas as pd

class serviceClustering(object):

    def __init__(self, dataSet, pathResponse, sizeEval, percentage, significancia):

        self.sizeEval = sizeEval
        self.dataSet = dataSet#conjunto de datos escalados y pre procesados
        self.pathResponse = pathResponse
        self.percentage = percentage
        self.significancia = significancia
        self.applyClustering = processClustering.aplicateClustering(self.dataSet)

    #metodo que permite hacer la ejecucion del servicio...
    def execProcess(self):

        header = ["algorithm", "params", "groups", "memberG1", "memberG2", "calinski_harabaz_score", "silhouette_score"]
        responseProcess = []
        logResponsesError = []
        indexResponse = []
        indexResponseError = []

        contIndex = 0
        contIndexError = 0

        k=2
        responseExec = self.applyClustering.aplicateKMeans(k)#se aplica el algoritmo...

        if responseExec == 0:#ok!
            params = "K = %d" % k
            result = evaluationClustering.evaluationClustering(self.dataSet, self.applyClustering.labels)#evaluamos...
            numberGroups = len(list(set(self.applyClustering.labels)))

            label1, label2 = self.checkMembersDistributionCluster(self.applyClustering.labels)
            response = ["K-Means", params, numberGroups, label1, label2, result.calinski, result.siluetas]
            responseProcess.append(response)
            contIndex+=1
            indexResponse.append(contIndex)

        else:
            message = "Error exec K-Means with K %d" % k
            logResponsesError.append(message)
            contIndexError+=1
            indexResponseError.append(contIndexError)

        #aplicamos Birch
        responseExec = self.applyClustering.aplicateBirch(k)#se aplica el algoritmo...

        if responseExec == 0:
            result = evaluationClustering.evaluationClustering(self.dataSet, self.applyClustering.labels)#evaluamos...
            params = "K = %d" % k
            numberGroups = len(list(set(self.applyClustering.labels)))
            label1, label2 = self.checkMembersDistributionCluster(self.applyClustering.labels)
            response = ["Birch", params, numberGroups, label1, label2, result.calinski, result.siluetas]
            responseProcess.append(response)
            contIndex+=1
            indexResponse.append(contIndex)
        else:
            message = "Error exec Birch with K %d" % k
            logResponsesError.append(message)
            contIndexError+=1
            indexResponseError.append(contIndexError)

        #aplicamos AgglomerativeClustering
        for affinity in ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']:
            for linkage in ['ward', 'complete', 'average', 'single']:
                responseExec = self.applyClustering.aplicateAlgomerativeClustering(linkage, affinity, k)#se aplica el algoritmo...

                params = "affinity = %s linkage = %s k = %d" % (affinity, linkage, k)
                if responseExec == 0:
                    result = evaluationClustering.evaluationClustering(self.dataSet, self.applyClustering.labels)#evaluamos...
                    numberGroups = len(list(set(self.applyClustering.labels)))
                    label1, label2 = self.checkMembersDistributionCluster(self.applyClustering.labels)
                    response = ["AgglomerativeClustering", params, numberGroups, label1, label2, result.calinski, result.siluetas]
                    responseProcess.append(response)
                    contIndex+=1
                    indexResponse.append(contIndex)
                else:
                    message = "Error exec AgglomerativeClustering with params %s" % params
                    logResponsesError.append(message)
                    contIndexError+=1
                    indexResponseError.append(contIndexError)

        #exportamos el resultado en formato dataframe
        self.dataFrame = pd.DataFrame(responseProcess, columns=header, index=indexResponse)
        self.dataFrameLog = pd.DataFrame(logResponsesError, columns=["Message Error"], index = indexResponseError)
        #self.dataFrame.to_csv(self.pathResponse+"ResponseProcess_Job_Clustering.csv", index=indexResponse)
        #self.dataFrameLog.to_csv(self.pathResponse+"ResponseProcess_Job_Clustering_Error.csv", index=indexResponseError)

        #generamos el resumen del proceso
        summary = summaryScan.summaryProcessClusteringScan(self.dataFrame, self.pathResponse+"ResponseProcess_Job_Clustering.csv", self.pathResponse)

        #revisamos el resultado y evaluamos si se puede seguir dividiendo
        if len(responseProcess)>0:

            checkData = checkProcessCluster.checkProcess(self.dataFrame)
            rowValues = []
            for key in self.dataFrame:
                rowValues.append(list(self.dataFrame[key])[checkData.candidato])

            #evaluamos que sucede con la informacion, el 5 implica que supere el 5% de la totalidad la muestra de datos
            if checkData.checkSplitter(rowValues[3], rowValues[4], self.percentage, self.sizeEval) == 1:
                #ejecutamos el cluster y formamos los data set con las divisiones
                if rowValues[0] == "K-Means":
                    self.applyClustering.aplicateKMeans(2)#se aplica el algoritmo...

                elif rowValues[0] == "AgglomerativeClustering":
                    #obtenemos el linkage y el affinity
                    params = rowValues[1].split(" ")
                    affinity = params[2]
                    linkage = params[5]
                    self.applyClustering.aplicateAlgomerativeClustering(linkage, affinity, 2)#se aplica el algoritmo...
                else:
                    self.applyClustering.aplicateBirch(2)

                #formamos los dataframe con los conjuntos de datos generados
                matrixGroup1 = []
                matrixGroup2 = []

                for i in range(len(self.applyClustering.labels)):
                    row = []
                    for key in self.dataSet:
                        row.append(self.dataSet[key][i])
                    if self.applyClustering.labels[i] == 0:
                        matrixGroup1.append(row)
                    else:
                        matrixGroup2.append(row)

                #tomamos las matrices y hacemos los analisis estadistico con el fin de poder corroborar su validez
                responseStatistic = checkData.checkStatisticValidation(matrixGroup1, matrixGroup2, self.significancia)

                if responseStatistic == 1:
                    #formamos los dataFrame y exportamos los resultados
                    header = []
                    for key in self.dataSet:
                        header.append(key)

                    dataG1 = pd.DataFrame(matrixGroup1, columns=header)
                    dataG2 = pd.DataFrame(matrixGroup2, columns=header)

                    #dataG1.to_csv(self.pathResponse+"group1.csv", index=False)
                    #dataG2.to_csv(self.pathResponse+"group2.csv", index=False)
                    return [1,dataG1,dataG2] #podemos seguir dividiendo, retorno los grupos
                    #return 1#podemos seguir dividiendo
                else:
                    return [-1,-1,-1]#no se puede seguir dividiendo
            else:
                return [-1,-1,-1]#no se puede seguir dividiendo
        else:
            return [-1,-1,-1] #no se puede seguir dividiendo

    #funcion que permite poder contar los elementos de la clase o categoria indicada
    def checkMembersDistributionCluster(self, labels):

        label1 = 0
        label2 = 0

        for label in labels:
            if label == 0:
                label1+=1
            else:
                label2+=1

        return label1, label2
