'''
script que recibe un set de datos y retorna en formato json la lista de valores asociados a la key del data set asociado
a la respuesta...
'''

import json
import pandas as pd

class searchData(object):

    def __init__(self, dataSet, key):

        self.dataSet = dataSet
        self.key = key
        self.dataValues = pd.read_csv(self.dataSet)

    #metodo que permite hacer la busqueda de los elementos...
    def searchValues(self):

        dictData = {}
        values = []
        for i in range (len(self.dataSet)):
            values.append(self.dataSet[self.key][i])

        dictData.update({"response":values})
        print json.dumps(dictData)

    def searchValuesInData(self):

        values = []
        for i in range(len(self.dataValues)):
            values.append(self.dataValues[self.key][i])
        return values
