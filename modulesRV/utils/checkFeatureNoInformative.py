import pandas as pd

class checkNonInformativeFeatures(object):

    def __init__(self, dataSet, nameFileExport):

        self.dataSet = dataSet
        self.nameFileExport = nameFileExport

    def evalNonInformative(self):

        self.listRemoveFeatures = []
        print "dentro del metodo"
        for key in self.dataSet:
            data = self.dataSet[key]

            dataUnique = list(set(data))

            if len(dataUnique) ==1:
                self.listRemoveFeatures.append(key)
                del self.dataSet[key]
        self.dataSet.to_csv(self.nameFileExport, index=False)
