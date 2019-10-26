'''
clase que tiene la responsabilidad de obtener los outliers de un arreglo, con respecto a sus
diferentes valores de desviacion estandar
'''

import numpy as np

class getOutliers(object):

    def __init__(self, arrayData):

        self.arrayData = arrayData

    #metodo que permite obtener los outliers de la data con respecto a sus desviaciones...
    def getOutliersNormalValues(self):

        #calculamos media y desviacion estandar
        mean = np.mean(self.arrayData)
        std = np.std(self.arrayData)

        #formamos el diccionario con la respuesta...
        dictResponse = {"mean": mean, "std":std}

        #valores a 1 desviacion estandar
        oneDesviacion = self.getCountData(std, mean)

        #valores a 1.5 desviacion estandar
        one_mediosDesviacion = self.getCountData(std*1.5, mean)

        #valores a 2 desviaciones estandar
        twoDesviation = self.getCountData(std*2, mean)

        #valores a 3 desviaciones estandar
        threeDesviation = self.getCountData(std*3, mean)

        #formamos el diccionario
        dictResponse = {"mean": mean, "std":std, "one": {"positive": oneDesviacion[0], "negative": oneDesviacion[1]}, "one_media": {"positive": one_mediosDesviacion[0], "negative": one_mediosDesviacion[1]}, "two": {"positive": twoDesviation[0], "negative": twoDesviation[1]}, "three": {"positive": threeDesviation[0], "negative": threeDesviation[1]}}

        return dictResponse

    #obtener valores segun cantidad de desviaciones, positivos y negativos
    def getCountData(self, std, mean):

        countPos = 0
        countNeg = 0

        positiveValue = mean + std
        negativeValue = mean - std
        for element in self.arrayData:
            if element >=positiveValue:
                countPos+=1
            if element <= negativeValue:
                countNeg+=1

        return countPos, countNeg
