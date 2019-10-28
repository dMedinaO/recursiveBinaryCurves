from sklearn.metrics import accuracy_score

class evalTraining(object):

    def __init__(self, classifiers, dataSetTesting, classTesting):
        self.classifiers = classifiers
        self.dataSetTesting = dataSetTesting
        self.classTesting = classTesting

        self.valuesPredic = self.classifiers.predict(self.dataSetTesting)#generamos los valores de prediccion

    #funcion que permite hacer la evaluacion del conjunto de datos
    def getPerformanceValues(self):

        print self.valuesPredic
        print self.classTesting
        accuracy = accuracy_score(self.classTesting, self.valuesPredic)
        print accuracy
