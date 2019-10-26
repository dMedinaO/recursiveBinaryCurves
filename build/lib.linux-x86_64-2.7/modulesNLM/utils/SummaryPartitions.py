import pandas as pd
import json
from modulesNLM.clustering_analysis import evaluationClustering
#utils para el manejo de set de datos y su normalizacion
from modulesNLM.utils import transformDataClass
from modulesNLM.utils import transformFrequence
from modulesNLM.utils import ScaleNormalScore
from modulesNLM.utils import ScaleMinMax
from modulesNLM.utils import ScaleDataSetLog
from modulesNLM.utils import ScaleLogNormalScore

from modulesNLM.utils import summaryScanProcess
from modulesNLM.utils import encodingFeatures

class summaryPartitions(object):

    def __init__(self, pathPartitions, numberPartitions):

        self.pathPartitions = pathPartitions
        self.numberPartitions = numberPartitions

    #metodo que permite obtener el largo de las particiones generadas
    def getLengthPartitions(self):

        self.arrayLength = []
        for i in range(1, self.numberPartitions+1):
            dataSet = pd.read_csv(self.pathPartitions+"p"+str(i)+"/"+"p"+str(i)+".csv")
            self.arrayLength.append(len(dataSet))

    #metodo que permite crear un conjunto de datos con los elementos en cada particion y adiciona la columna partition
    def createDataSetWithPartitionID(self):

        matrixData = []
        for i in range(1, self.numberPartitions+1):
            dataSet = pd.read_csv(self.pathPartitions+"p"+str(i)+"/"+"p"+str(i)+".csv")

            #recorremos los elementos para completar la matriz
            for j in range(len(dataSet)):
                row = []

                for key in dataSet:
                    row.append(dataSet[key][i])
                row.append(i)
                matrixData.append(row)

        #formamos el header
        header = []
        dataSet = pd.read_csv(self.pathPartitions+"p"+str(i)+"/"+"p"+str(i)+".csv")
        for key in dataSet:
            header.append(key)
        header.append("partition")

        #formamos el dataframe y exportamos el conjunto de datos a un archivo csv
        self.dataFrame = pd.DataFrame(matrixData, columns=header)
        self.dataFrame.to_csv(self.pathPartitions+"dataSetWithPartitions.csv", index=False)

    #metodo que permite evaluar los cluster generados
    def getMeasuresCluster(self):

        #procesamos el set de datos para obtener los atributos y las clases...
        columnas=self.dataFrame.columns.tolist()
        x=columnas[len(columnas)-1]
        targetResponse=self.dataFrame[x]#clases
        y=columnas[0:len(columnas)-1]
        dataValues=self.dataFrame[y]#atributos

        encoding = encodingFeatures.encodingFeatures(dataValues, 20)
        encoding.evaluEncoderKind()
        dataSetNewFreq = encoding.dataSet

        #ahora aplicamos el procesamiento segun lo expuesto
        applyNormal = ScaleNormalScore.applyNormalScale(dataSetNewFreq)
        data = applyNormal.dataTransform

        self.resultEvaluation = evaluationClustering.evaluationClustering(data, targetResponse)#evaluamos...

    #metodo que permite formar el JSON con los resumenes
    def createJSONSummary(self):

        dictResponse = {}
        dictResponse.update({"calinski_harabaz_score":self.resultEvaluation.calinski})
        dictResponse.update({"silhouette_score":self.resultEvaluation.siluetas})
        dictResponse.update({"partitions_length":self.arrayLength})

        #exportamos a JSON
        nameFileExport = self.pathPartitions+"summaryPartitions.json"
        with open(nameFileExport, 'w') as fp:
            json.dump(dictResponse, fp)
