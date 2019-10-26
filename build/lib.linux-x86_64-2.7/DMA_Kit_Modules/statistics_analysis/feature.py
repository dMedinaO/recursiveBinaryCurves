'''
clase que representa una caracteristica
'''

from modulesProject.statistics_analysis import getValuesInDataSet

class feature(object):

    #constructor de la clase
    def __init__(self, nameData, kindData):

        self.nameData = nameData
        self.kindData = kindData

    #metodo que permite obtener los valores asociados...
    def getValuesInDataSet(self, dataSet):

        searching = getValuesInDataSet.searchData(dataSet, self.nameData)
        self.data = searching.searchValuesInData()
