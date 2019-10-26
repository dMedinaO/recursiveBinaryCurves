'''
clase con la responsabilidad de obtener todos los atributos, su tipo y sus valores...
'''

import pandas as pd
from modulesProject.statistics_analysis import feature
from modulesProject.dataBase_module import ConnectDataBase
from modulesProject.dataBase_module import HandlerQuery

class processFeatureInDataSet(object):

    def __init__(self, dataSet, idDataSet):

        self.dataSet = dataSet
        self.idDataSet = idDataSet
        self.listFeatures = []#lista de las caracteristicas...

        #instancias a la base de datos...
        self.connect = ConnectDataBase.ConnectDataBase()
        self.handler = HandlerQuery.HandlerQuery()

    #process data.
    def processDataValues(self):

        #get query and kind
        query = "select feature.nameFeature, feature.kind from feature where feature.dataSet = %s" %self.idDataSet

        self.connect.initConnectionDB()
        listResponse = self.handler.queryBasicDataBase(query, self.connect)

        #hacemos la instancia y las insertamos en la lista...
        for element in listResponse:
            featureObject = feature.feature(str(element[0]), str(element[1]))
            self.listFeatures.append(featureObject)

        self.connect.closeConnectionDB()

        #obtenemos los valores de la lista de features...
        for i in range(len(self.listFeatures)):
            self.listFeatures[i].getValuesInDataSet(self.dataSet)
