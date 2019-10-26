import pandas as pd

class processParams(object):

    def __init__(self, pathExport, listPerformance):

        self.pathExport = pathExport
        self.listPerformance = listPerformance

    #funcion que permite formar un diccionario con respecto a los parametros dependiendo del tipo de algoritmo
    def getDictParamsData(self, paramsValues):

        dictResponse = {}

        listParams = paramsValues.split("-")
        for paramsData in listParams:
            data = paramsData.split(":")
            try:
                dictResponse.update({data[0]:data[1]})
            except:
                dictResponse.update({"paramsvalue": "Default"})
        return dictResponse

    #funcion que permite obtener el mejor modelo en cada medida
    def getBestModels(self):

        self.listModels = []

        for measure in self.listPerformance:

            dictMeasure = {}
            dataModels = pd.read_csv(self.pathExport+measure+"_ranking.csv")
            algorithm = dataModels['Algorithm'][0]
            params = dataModels['Params'][0]
            dictMeasure.update({"algorithm":algorithm})

            for measureV in self.listPerformance:

                dictMeasure.update({measureV:dataModels[measureV][0]})

            #agregamos la informacion del diccionario de los parametros
            dictMeasure.update({"params":self.getDictParamsData(params)})
            self.listModels.append(dictMeasure)
