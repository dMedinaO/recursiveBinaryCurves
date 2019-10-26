########################################################################
# callService.py,
#
# Execute the clustering service using job process, for wich a service call in background need to be done.
# All clustering process will be addressed with their own parameter and answers.
#
# Receives a normalized dataset, job id and user id. The data will be used to save information in user's personal path.
#
# Returns information as an csv file, calinski medition and sihouette through two histogram. Finally a json file with the process resume.
#
# NOTE: in the clustering service all methods are tested without the user confirmation. Thus this service implementation is easier than the supervised learning service
#
# python modules in the directory of the same, in order to be able to be
# recognized from any python call and be indexed within the library itself.
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

from modulesNLM.clustering_analysis import processClustering
from modulesNLM.clustering_analysis import evaluationClustering
from modulesNLM.clustering_analysis import summaryScan

from modulesNLM.utils import transformFrequence
from modulesNLM.utils import ScaleNormalScore
from modulesNLM.utils import ScaleMinMax
from modulesNLM.utils import ScaleDataSetLog
from modulesNLM.utils import ScaleLogNormalScore
from modulesNLM.utils import encodingFeatures
from modulesNLM.checks_module import checkProcessCluster

import pandas as pd

class serviceClustering(object):

    def __init__(self, dataSet, pathResponse, optionNormalize, featureClass, kindDataSet, threshold, sizeEval):

        self.sizeEval = sizeEval
        self.dataOriginal = dataSet#matriz de elementos
        self.featureClass = featureClass#nombre de la columna respuesta
        self.kindDataSet = kindDataSet#tipo de set de datos: clasificacion/regresion
        self.optionNormalize = optionNormalize
        self.threshold = threshold#umbral de desbalance aceptado
        self.processDataSet()#hacemos el preprocesamiento a los datos
        self.pathResponse = pathResponse
        self.applyClustering = processClustering.aplicateClustering(self.dataSet)

    #metodo que permite procesar el set de datos segun la opcion del usuario a normalizar
    def processDataSet(self):

        self.dataResponse = self.dataOriginal[self.featureClass]#obtenemos la variable respuesta

        dictData = {}

        for key in self.dataOriginal:
            if key != self.featureClass:
                arrayFeature = []
                for i in self.dataOriginal[key]:
                    arrayFeature.append(i)
                dictData.update({key:arrayFeature})

        #formamos el nuevo set de datos...
        self.dataSet = pd.DataFrame(dictData)

        #codificacion de variables categoricas
        encoding = encodingFeatures.encodingFeatures(self.dataSet, 20)
        encoding.evaluEncoderKind()
        dataSetNewFreq = encoding.dataSet

        #ahora aplicamos el procesamiento segun lo expuesto
        if self.optionNormalize == 1:#normal scale
            applyNormal = ScaleNormalScore.applyNormalScale(dataSetNewFreq)
            self.dataSet = applyNormal.dataTransform

        if self.optionNormalize == 2:#min max scaler
            applyMinMax = ScaleMinMax.applyMinMaxScaler(dataSetNewFreq)
            self.dataSet = applyMinMax.dataTransform

        if self.optionNormalize == 3:#log scale
            applyLog = ScaleDataSetLog.applyLogScale(dataSetNewFreq)
            self.dataSet = applyLog.dataTransform

        if self.optionNormalize == 4:#log normal scale
            applyLogNormal = ScaleLogNormalScore.applyLogNormalScale(dataSetNewFreq)
            self.dataSet = applyLogNormal.dataTransform

    #metodo que permite hacer la ejecucion del servicio...
    def execProcess(self):

        header = ["algorithm", "params", "groups", "memberG1", "memberG2", "calinski_harabaz_score", "silhouette_score"]
        responseProcess = []
        logResponsesError = []
        indexResponse = []
        indexResponseError = []

        contIndex = 0
        contIndexError = 0

        print "Process K-Means"
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
        print "Process Birch"
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
        print "Process AgglomerativeClustering"
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
        self.dataFrame.to_csv(self.pathResponse+"ResponseProcess_Job_Clustering.csv", index=indexResponse)
        self.dataFrameLog.to_csv(self.pathResponse+"ResponseProcess_Job_Clustering_Error.csv", index=indexResponseError)

        #generamos el resumen del proceso
        summary = summaryScan.summaryProcessClusteringScan(self.dataFrame, self.pathResponse+"ResponseProcess_Job_Clustering.csv", self.pathResponse)
        #summary.createHistogram()
        #summary.createRankingFile()
        #summary.createStatisticSummary()

        #chequeamos el procesos de clustering y entregamos la informacion
        if len(responseProcess)>0:
            checkData = checkProcessCluster.checkProcess(self.dataFrame)

            #dado el candidato obtenemos los valores de los elementos
            #header = ["algorithm", "params", "groups", "memberG1", "memberG2", "calinski_harabaz_score", "silhouette_score"]
            rowValues = []
            for key in self.dataFrame:
                rowValues.append(list(self.dataFrame[key])[checkData.candidato])

            #evaluamos que sucede con la informacion, el 5 implica que supere el 5% de la totalidad la muestra de datos
            if checkData.checkSplitter(rowValues[3], rowValues[4], 5, self.sizeEval) == 1:
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
                    for key in self.dataOriginal:
                        row.append(self.dataOriginal[key][i])
                    if self.applyClustering.labels[i] == 0:
                        matrixGroup1.append(row)
                    else:
                        matrixGroup2.append(row)

                #formamos los dataFrame y exportamos los resultados
                header = []
                for key in self.dataOriginal:
                    header.append(key)

                dataG1 = pd.DataFrame(matrixGroup1, columns=header)
                dataG2 = pd.DataFrame(matrixGroup2, columns=header)

                if self.kindDataSet == 1:#el set de datos corresponde a clasificacion
                    #evaluamos el desbalance de clases en cada conjunto de datos generados
                    responseUG1 = checkData.checkEvalClass(dataG1[self.featureClass], self.threshold)
                    responseUG2 = checkData.checkEvalClass(dataG2[self.featureClass], self.threshold)

                    if responseUG1 ==0 and responseUG2 ==0:
                        dataG1.to_csv(self.pathResponse+"group1.csv", index=False)
                        dataG2.to_csv(self.pathResponse+"group2.csv", index=False)

                        return 1#podemos seguir dividiendo
                    else:
                        print "Error 0"
                        return -1#se genero un desbalance de clases
                else:
                    dataG1.to_csv(self.pathResponse+"group1.csv", index=False)
                    dataG2.to_csv(self.pathResponse+"group2.csv", index=False)
                    return 1#podemos seguir dividiendo
            else:
                print "Error I"
                return -1#no se puede seguir dividiendo
        else:
            print "Error II"
            return -1#no se puede seguir dividiendo
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
