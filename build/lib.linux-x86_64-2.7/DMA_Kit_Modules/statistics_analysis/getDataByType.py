'''
clase con la responsabilidad de obtener los datos del tipo continuo y formar el json que se requiere para la carga de
datos en la vista...
tiene diferentes metodos con respecto al tipo escala que debe aplicarse
'''

from modulesProject.statistics_analysis import getFeatures
from modulesProject.statistics_analysis import getOutliers

import json

class dataByType(object):

    def __init__(self, headerFeatures):

        self.headerFeatures = headerFeatures

    #metodo que permite obtener los datos continuos del set de datos
    def getValuesContinues(self):

        keys = []
        features = []
        outliers = []

        for feature in self.headerFeatures.listFeatures:
            if feature.kindData == "CONTINUA":
                keys.append(feature.nameData)
                features.append(feature.data)
                #obtenemos los outliers para el atributo
                outlierObject = getOutliers.getOutliers(feature.data)
                outliers.append(outlierObject.getOutliersNormalValues())

        dictResponse = {}
        dictResponse.update({"keys": keys})
        dictResponse.update({"data": features})
        dictResponse.update({"outliers": outliers})

        return dictResponse

    #metodo que permit eobtener los datos discretos del set de datos...
    def getValuesDiscrete(self):

        dictResponse = {}

        for feature in self.headerFeatures.listFeatures:
            if feature.kindData == "DISCRETA":

                #obtenemos: los valores unicos y la frecuencia de ellos
                uniqueValues = list(set(feature.data))
                arrayFrequence = []

                #contamos los valores unicos...
                for element in uniqueValues:
                    cont=0
                    for i in range(len(feature.data)):
                        if element == feature.data[i]:
                            cont+=1
                    arrayFrequence.append(cont)

                frequenceResponse = {"keyData": uniqueValues, "valueFrequence" : arrayFrequence}
                dictResponse.update({feature.nameData:frequenceResponse})
        return dictResponse
