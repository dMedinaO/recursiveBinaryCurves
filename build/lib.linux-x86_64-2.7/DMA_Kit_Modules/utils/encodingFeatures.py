import pandas as pd

class encodingFeatures(object):

    def __init__(self, dataSet, treshold):

        self.dataSet = dataSet

        if treshold == "DEFAULT":
            self.treshold = 20
        else:
            try:
                self.treshold = int(treshold)
            except:
                self.treshold = 20
                pass

        self.evalData()

    #funcion que permite implementar ordinal encoder
    def ordinalEncoderData(self, dictValues):

        for key in dictValues:
            for i in range(len(self.dataSet)):
                self.dataSet[key][i] = dictValues[key][self.dataSet[key][i]]

    #funcion que permite implementar one hot Encoder
    def oneHotEncoderData(dictValues, dataSet):

        matrix = []

        for index in range(len(dataSet)):
            row = []
            for key in dictValues:
                #obtenemos los valores del diccionario con dicha key
                for values in dictValues[key]:
                    if dataSet[key][index] == values:
                        row.append(1)
                    else:
                        row.append(0)
            matrix.append(row)

        return matrix

    #funcion para evaluar la data
    def evalData(self):

        self.numberCategoryData = 0
        self.countKey = 0
        self.keyCategoric = []
        self.newFeatures = 0

        for key in self.dataSet:

            response=0
            for element in self.dataSet[key]:
                try:
                    data = float(element)
                except:
                    response=1
                    self.keyCategoric.append(key)
                    break
            if response == 1:
                self.numberCategoryData+=1
            self.countKey+=1

        self.matrixDict = {}
        #obtenemos la cantidad de variables unicas a agregar
        for key in self.keyCategoric:
            values = list(set(self.dataSet[key]))
            #formamos un diccionario con los valores de las listas
            dictData = {}
            for i in range(len(values)):

                dictData.update({values[i]:i})
            self.matrixDict.update({key:dictData})
            self.newFeatures+=len(values)

    #metodo para decidir que utilizar
    def evaluEncoderKind(self):

        #evaluamos el umbral...
        self.dataValue = float(self.newFeatures)*100/float(self.countKey)

        if self.dataValue>self.treshold:#ordinal encoder
            self.ordinalEncoderData(self.matrixDict)
        else:#
            matrixAdd = self.oneHotEncoderData(self.matrixDict, self.dataSet)

            for i in range(len(matrixAdd[0])):
                column = []
                for j in range(len(matrixAdd)):
                    column.append(matrixAdd[j][i])
                feature = "feature_"+str(i)
                self.dataSet[feature] = column

            #por cada columna en el diccionario la eliminamos
            for key in matrixDict:
                self.dataSet.drop([key], axis='columns', inplace=True)
