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

from DMA_Kit_Modules.clustering_analysis import processClustering
from DMA_Kit_Modules.clustering_analysis import evaluationClustering
from DMA_Kit_Modules.clustering_analysis import summaryScan

from DMA_Kit_Modules.utils import transformFrequence
from DMA_Kit_Modules.utils import ScaleNormalScore
from DMA_Kit_Modules.utils import ScaleMinMax
from DMA_Kit_Modules.utils import ScaleDataSetLog
from DMA_Kit_Modules.utils import ScaleLogNormalScore

import pandas as pd

class serviceClustering(object):

    def __init__(self, dataSet, pathResponse, optionNormalize):

        self.optionNormalize = optionNormalize
        self.processDataSet(dataSet)#hacemos el preprocesamiento a los datos
        self.pathResponse = pathResponse
        self.applyClustering = processClustering.aplicateClustering(self.dataSet)

    #metodo que permite procesar el set de datos segun la opcion del usuario a normalizar
    def processDataSet(self, dataSetInput):

        #ahora transformamos el set de datos por si existen elementos discretos...
        transformDataSet = transformFrequence.frequenceData(dataSetInput)
        dataSetNewFreq = transformDataSet.dataTransform

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

        header = ["algorithm", "params", "groups", "calinski_harabaz_score", "silhouette_score"]
        responseProcess = []
        logResponsesError = []
        indexResponse = []
        indexResponseError = []

        contIndex = 0
        contIndexError = 0

        print "Process K-Means"
        #aplicamos k-means variando el numero de k
        for k in range (2, 10):
            responseExec = self.applyClustering.aplicateKMeans(k)#se aplica el algoritmo...

            if responseExec == 0:#ok!
                params = "K = %d" % k
                result = evaluationClustering.evaluationClustering(self.dataSet, self.applyClustering.labels)#evaluamos...
                numberGroups = len(list(set(self.applyClustering.labels)))
                response = ["K-Means", params, numberGroups, result.calinski, result.siluetas]
                responseProcess.append(response)
                contIndex+=1
                indexResponse.append(contIndex)

            else:
                message = "Error exec K-Means with K %d" % k
                logResponsesError.append(message)
                contIndexError+=1
                indexResponseError.append(contIndexError)

        #aplicamos MeanShift...
        print "Process MeanShift"
        responseExec = self.applyClustering.aplicateMeanShift()#se aplica el algoritmo...

        if responseExec == 0:
            result = evaluationClustering.evaluationClustering(self.dataSet, self.applyClustering.labels)#evaluamos...
            numberGroups = len(list(set(self.applyClustering.labels)))
            response = ["MeanShift", "Default", numberGroups, result.calinski, result.siluetas]
            responseProcess.append(response)
            contIndex+=1
            indexResponse.append(contIndex)
        else:
            message = "Error exec MeanShift"
            logResponsesError.append(message)
            contIndexError+=1
            indexResponseError.append(contIndexError)

        #aplicamos DBSCAN
        print "Process DBSCAN"
        responseExec = self.applyClustering.aplicateDBSCAN()#se aplica el algoritmo...

        if responseExec == 0:
            result = evaluationClustering.evaluationClustering(self.dataSet, self.applyClustering.labels)#evaluamos...
            numberGroups = len(list(set(self.applyClustering.labels)))
            response = ["DBSCAN", "Default", numberGroups, result.calinski, result.siluetas]
            responseProcess.append(response)
            contIndex+=1
            indexResponse.append(contIndex)
        else:
            message = "Error exec DBSCAN"
            logResponsesError.append(message)
            contIndexError+=1
            indexResponseError.append(contIndexError)

        #aplicamos aplicateAffinityPropagation
        print "Process AffinityPropagation"
        responseExec = self.applyClustering.aplicateAffinityPropagation()#se aplica el algoritmo...

        if responseExec == 0:
            result = evaluationClustering.evaluationClustering(self.dataSet, self.applyClustering.labels)#evaluamos...
            numberGroups = len(list(set(self.applyClustering.labels)))
            response = ["AffinityPropagation", "Default", numberGroups, result.calinski, result.siluetas]
            responseProcess.append(response)
            contIndex+=1
            indexResponse.append(contIndex)
        else:
            message = "Error exec AffinityPropagation"
            logResponsesError.append(message)
            contIndexError+=1
            indexResponseError.append(contIndexError)

        #aplicamos Birch
        print "Process Birch"
        for k in range (2, 10):
            responseExec = self.applyClustering.aplicateBirch(k)#se aplica el algoritmo...

            if responseExec == 0:
                result = evaluationClustering.evaluationClustering(self.dataSet, self.applyClustering.labels)#evaluamos...
                params = "K = %d" % k
                numberGroups = len(list(set(self.applyClustering.labels)))
                response = ["Birch", params, numberGroups, result.calinski, result.siluetas]
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
        for k in range (2, 10):
            for affinity in ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']:
                for linkage in ['ward', 'complete', 'average', 'single']:

                    responseExec = self.applyClustering.aplicateAlgomerativeClustering(linkage, affinity, k)#se aplica el algoritmo...

                    if responseExec == 0:
                        result = evaluationClustering.evaluationClustering(self.dataSet, self.applyClustering.labels)#evaluamos...
                        params = "affinity = %s linkage = %s k = %d" % (affinity, linkage, k)
                        numberGroups = len(list(set(self.applyClustering.labels)))
                        response = ["AgglomerativeClustering", params, numberGroups, result.calinski, result.siluetas]
                        responseProcess.append(response)
                        contIndex+=1
                        indexResponse.append(contIndex)
                    else:
                        message = "Error exec AgglomerativeClustering with params %s" % params
                        logResponsesError.append(message)
                        contIndexError+=1
                        indexResponseError.append(contIndexError)

        #exportamos el resultado en formato dataframe
        dataFrame = pd.DataFrame(responseProcess, columns=header, index=indexResponse)
        dataFrameLog = pd.DataFrame(logResponsesError, columns=["Message Error"], index = indexResponseError)
        dataFrame.to_csv(self.pathResponse+"ResponseProcess_Job_Clustering.csv")
        dataFrameLog.to_csv(self.pathResponse+"ResponseProcess_Job_Clustering_Error.csv")

        #generamos el resumen del proceso
        summary = summaryScan.summaryProcessClusteringScan(dataFrame, self.pathResponse+"ResponseProcess_Job_Clustering.csv", self.pathResponse)
        summary.createHistogram()
        summary.createRankingFile()
        summary.createStatisticSummary()
