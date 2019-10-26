########################################################################
# execAlgorithm.py,
#
# Executes supervised learning algorithm to create model training.
# REceiver unique model's parameters and executes the complements for the outputs generated.
#
# Algorithms:
#
# 1 Adaboost
# 2 Bagging
# 3 Bernoulli
# 4 Decision Tree
# 5 Gaussian
# 6 Gradient
# 7 KNN
# 8 MLP
# 9 NuSVC
# 10 RF
# 11 SVC
#
# Each one has differents parametesr for their implementations.
#
#
# Copyright (C) 2019  David Medina Ortiz, david.medina@cebib.cl
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301  USA
########################################################################

#importamos los algoritmos...
from DMA_Kit_Modules.supervised_learning_analysis import AdaBoost
from DMA_Kit_Modules.supervised_learning_analysis import Baggin
from DMA_Kit_Modules.supervised_learning_analysis import BernoulliNB
from DMA_Kit_Modules.supervised_learning_analysis import DecisionTree
from DMA_Kit_Modules.supervised_learning_analysis import GaussianNB
from DMA_Kit_Modules.supervised_learning_analysis import Gradient
from DMA_Kit_Modules.supervised_learning_analysis import knn
from DMA_Kit_Modules.supervised_learning_analysis import MLP
from DMA_Kit_Modules.supervised_learning_analysis import NuSVM
from DMA_Kit_Modules.supervised_learning_analysis import RandomForest
from DMA_Kit_Modules.supervised_learning_analysis import SVM

#importamos los metodos para generar el resto de los resultados
from DMA_Kit_Modules.supervised_learning_analysis import createConfusionMatrix
from DMA_Kit_Modules.supervised_learning_analysis import createLearningCurve
from DMA_Kit_Modules.utils import encodingFeatures

#metodos de la libreria utils...
from DMA_Kit_Modules.utils import transformDataClass
from DMA_Kit_Modules.utils import transformFrequence
from DMA_Kit_Modules.utils import ScaleNormalScore
from DMA_Kit_Modules.utils import ScaleMinMax
from DMA_Kit_Modules.utils import ScaleDataSetLog
from DMA_Kit_Modules.utils import ScaleLogNormalScore

import pandas as pd
import json

class execAlgorithm(object):

    #constructor de la clase
    def __init__(self, dataSet, pathResponse, algorithm, params, validation, featureClass, optionNormalize, treshold):

        self.dataSet = dataSet
        self.pathResponse = pathResponse
        self.algorithm = algorithm
        self.params = params#params es una lista de parametros asociados al algoritmo
        self.validation = validation#validacion del algoritmo (valor de CV)
        self.featureClass = featureClass#el nombre del atributo que es la respuesta
        self.optionNormalize = optionNormalize#tipo de normalizacion a aplicar en el set de datos
        self.treshold = treshold#umbral para la codificacion de variables categoricas

        self.response = {}#diccionario con la respuesta para formar el json
        self.classArray = []#contiene el nombre de las clases

        #procesamos el data set para obtener las clases y los atributos
        self.createDataSet()
        self.response.update({'classTransform':self.dictTransform})

    #metodo que permite formar el set de datos y el target con la informacion...
    def createDataSet(self):

        targetResponse = self.dataSet[self.featureClass]
        dictData = {}

        for key in self.dataSet:
            if key != self.featureClass:
                arrayFeature = []
                for i in self.dataSet[key]:
                    arrayFeature.append(i)
                dictData.update({key:arrayFeature})

        #formamos el nuevo set de datos...
        dataSetNew = pd.DataFrame(dictData)

        #ahora evaluamos si la clase tiene valores discretos o continuos y los modificamos en caso de que sean discretos
        transformData = transformDataClass.transformClass(targetResponse)
        self.target = transformData.transformData
        self.dictTransform = transformData.dictTransform

        #formamos el class array...
        self.classArray = list(set(self.target))

        #ahora transformamos el set de datos por si existen elementos discretos...
        #transformDataSet = transformFrequence.frequenceData(dataSetNew)
        encoding = encodingFeatures.encodingFeatures(dataSetNew, self.treshold)
        encoding.evaluEncoderKind()
        dataSetNewFreq = encoding.dataSet

        #ahora aplicamos el procesamiento segun lo expuesto
        if self.optionNormalize == 1:#normal scale
            applyNormal = ScaleNormalScore.applyNormalScale(dataSetNewFreq)
            self.data = applyNormal.dataTransform

        elif self.optionNormalize == 2:#min max scaler
            applyMinMax = ScaleMinMax.applyMinMaxScaler(dataSetNewFreq)
            self.data = applyMinMax.dataTransform

        elif self.optionNormalize == 3:#log scale
            applyLog = ScaleDataSetLog.applyLogScale(dataSetNewFreq)
            self.data = applyLog.dataTransform

        else:#log normal scale
            applyLogNormal = ScaleLogNormalScore.applyLogNormalScale(dataSetNewFreq)
            self.data = applyLogNormal.dataTransform

    #metodo que permite evaluar la ejecucion del algoritmo con respecto a los parametros de entrada
    def execAlgorithmByOptions(self):

        if self.algorithm == 1:#Adaboost

            self.response.update({"algorithm": "AdaBoostClassifier"})
            paramsData = {}
            paramsData.update({"n_estimators": self.params[0]})
            paramsData.update({"algorithm": self.params[1]})
            self.response.update({"Params": paramsData})
            nameValidation = ""
            if self.validation == -1:
                nameValidation = "LeaveOneOut"
            else:
                nameValidation = str(self.validation)

            self.response.update({"Validation": "Cross Validation: " + nameValidation})

            #instancia al objeto...
            errorData = {}
            try:
                AdaBoostObject = AdaBoost.AdaBoost(self.data, self.target, int(self.params[0]), self.params[1], self.validation)
                if len(self.classArray)>2:
                    AdaBoostObject.trainingMethod(2)#multilabel
                else:
                    AdaBoostObject.trainingMethod(1)#binary
                performance = {}
                performance.update({"accuracy":AdaBoostObject.performanceData.scoreData[3]})
                performance.update({"recall": AdaBoostObject.performanceData.scoreData[4]})
                performance.update({"precision": AdaBoostObject.performanceData.scoreData[5]})
                performance.update({"f1": AdaBoostObject.performanceData.scoreData[6]})

                self.response.update({"Performance": performance})
                errorData.update({"exec_algorithm": "OK"})
            except:
                errorData.update({"exec_algorithm": "ERROR"})
                pass

            #trabajamos con la matriz de confusion
            try:
                #confusion matrix data
                confusionMatrixDemo = createConfusionMatrix.confusionMatrix(self.data, self.target, AdaBoostObject.model, self.validation, self.pathResponse, self.classArray)
                responseMatrix = confusionMatrixDemo.createConfusionMatrix(self.dictTransform)
                self.response.update({"matrixConfusionResponse":responseMatrix})
                errorData.update({"confusionMatrix" : "ok"})
            except:
                errorData.update({"confusionMatrix" : "error"})
                pass

            try:
                #learning curve
                learningCurveDemo = createLearningCurve.curveLearning(self.data, self.target, AdaBoostObject.model, self.validation, self.pathResponse)
                learningCurveDemo.createLearningCurve()
                errorData.update({"curveLearning" : "ok"})
            except:
                errorData.update({"curveLearning" : "error"})
                pass

            self.response.update({"errorExec": errorData})

            #exportamos tambien el resultado del json
            nameFile =self.pathResponse+"responseTraining.json"
            with open(self.pathResponse+"responseTraining.json", 'w') as fp:
                json.dump(self.response, fp)

        elif self.algorithm == 2:#Bagging

            self.response.update({"algorithm": "BaggingClassifier"})
            paramsData = {}
            paramsData.update({"n_estimators": int(self.params[0])})
            paramsData.update({"bootstrap": self.params[1]})
            self.response.update({"Params": paramsData})
            self.response.update({"Validation": "Cross Validation: " + str(self.validation)})

            errorData = {}
            try:
                #instancia al objeto...
                bagginObject = Baggin.Baggin(self.data,self.target,int(self.params[0]), self.params[1],self.validation)
                if len(self.classArray)>2:
                    bagginObject.trainingMethod(2)#multilabel
                else:
                    bagginObject.trainingMethod(1)#binary

                performance = {}
                performance.update({"accuracy":bagginObject.performanceData.scoreData[3]})
                performance.update({"recall": bagginObject.performanceData.scoreData[4]})
                performance.update({"precision": bagginObject.performanceData.scoreData[5]})
                performance.update({"f1": bagginObject.performanceData.scoreData[6]})

                self.response.update({"Performance": performance})

                errorData.update({"exec_algorithm": "OK"})
            except:
                errorData.update({"exec_algorithm": "ERROR"})
                pass

            try:

                #learning curve
                learningCurveDemo = createLearningCurve.curveLearning(self.data, self.target, bagginObject.model, self.validation, self.pathResponse)
                learningCurveDemo.createLearningCurve()
                errorData.update({"curveLearning" : "ok"})
            except:
                errorData.update({"curveLearning" : "error"})
                pass

            try:
                #confusion matrix data
                confusionMatrixDemo = createConfusionMatrix.confusionMatrix(self.data, self.target, bagginObject.model, self.validation, self.pathResponse, self.classArray)
                responseMatrix = confusionMatrixDemo.createConfusionMatrix(self.dictTransform)
                self.response.update({"matrixConfusionResponse":responseMatrix})
                errorData.update({"confusionMatrix" : "ok"})
            except:
                errorData.update({"confusionMatrix" : "error"})
                pass

            self.response.update({"errorExec": errorData})

            #exportamos tambien el resultado del json
            nameFile =self.pathResponse+"responseTraining.json"
            with open(self.pathResponse+"responseTraining.json", 'w') as fp:
                json.dump(self.response, fp)

        elif self.algorithm == 3:#Bernoulli

            self.response.update({"algorithm": "BernoulliNB"})
            paramsData = {}
            self.response.update({"Params": "Default"})
            self.response.update({"Validation": "Cross Validation: " + str(self.validation)})
            errorData = {}
            try:

                #instancia al objeto...
                bernoulliNB = BernoulliNB.Bernoulli(self.data, self.target, self.validation)
                if len(self.classArray)>2:
                    bernoulliNB.trainingMethod(2)#multilabel
                else:
                    bernoulliNB.trainingMethod(1)#binary

                performance = {}
                performance.update({"accuracy":bernoulliNB.performanceData.scoreData[3]})
                performance.update({"recall": bernoulliNB.performanceData.scoreData[4]})
                performance.update({"precision": bernoulliNB.performanceData.scoreData[5]})
                performance.update({"f1": bernoulliNB.performanceData.scoreData[6]})

                self.response.update({"Performance": performance})

                errorData.update({"exec_algorithm": "OK"})
            except:
                errorData.update({"exec_algorithm": "ERROR"})
                pass

            try:

                #learning curve
                learningCurveDemo = createLearningCurve.curveLearning(self.data, self.target, bernoulliNB.model, self.validation, self.pathResponse)
                learningCurveDemo.createLearningCurve()
                errorData.update({"curveLearning" : "ok"})
            except:
                errorData.update({"curveLearning" : "error"})
                pass

            try:
                #confusion matrix data
                confusionMatrixDemo = createConfusionMatrix.confusionMatrix(self.data, self.target, bernoulliNB.model, self.validation, self.pathResponse, self.classArray)
                responseMatrix = confusionMatrixDemo.createConfusionMatrix(self.dictTransform)
                self.response.update({"matrixConfusionResponse":responseMatrix})
                errorData.update({"confusionMatrix" : "ok"})
            except:
                errorData.update({"confusionMatrix" : "error"})
                pass

            self.response.update({"errorExec": errorData})

            #exportamos tambien el resultado del json
            nameFile =self.pathResponse+"responseTraining.json"
            with open(self.pathResponse+"responseTraining.json", 'w') as fp:
                json.dump(self.response, fp)

        elif self.algorithm == 4:#DecisionTree

            self.response.update({"algorithm": "DecisionTree"})
            paramsData = {}
            paramsData.update({"criterion": self.params[0]})
            paramsData.update({"splitter": self.params[1]})
            self.response.update({"Params": paramsData})
            self.response.update({"Validation": "Cross Validation: " + str(self.validation)})
            errorData = {}

            try:
                #instancia al objeto...
                decisionTreeObject = DecisionTree.DecisionTree(self.data, self.target, self.params[0], self.params[1],self.validation)
                if len(self.classArray)>2:
                    decisionTreeObject.trainingMethod(2)#multilabel
                else:
                    decisionTreeObject.trainingMethod(1)#binary

                performance = {}
                performance.update({"accuracy":decisionTreeObject.performanceData.scoreData[3]})
                performance.update({"recall": decisionTreeObject.performanceData.scoreData[4]})
                performance.update({"precision": decisionTreeObject.performanceData.scoreData[5]})
                performance.update({"f1": decisionTreeObject.performanceData.scoreData[6]})

                self.response.update({"Performance": performance})

                errorData.update({"exec_algorithm": "OK"})
            except:
                errorData.update({"exec_algorithm": "ERROR"})
                pass

            try:

                #learning curve
                learningCurveDemo = createLearningCurve.curveLearning(self.data, self.target, decisionTreeObject.model, self.validation, self.pathResponse)
                learningCurveDemo.createLearningCurve()
                errorData.update({"curveLearning" : "ok"})
            except:
                errorData.update({"curveLearning" : "error"})
                pass

            try:
                #confusion matrix data
                confusionMatrixDemo = createConfusionMatrix.confusionMatrix(self.data, self.target, decisionTreeObject.model, self.validation, self.pathResponse, self.classArray)
                responseMatrix = confusionMatrixDemo.createConfusionMatrix(self.dictTransform)
                self.response.update({"matrixConfusionResponse":responseMatrix})
                errorData.update({"confusionMatrix" : "ok"})
            except:
                errorData.update({"confusionMatrix" : "error"})
                pass

            self.response.update({"errorExec": errorData})

            #exportamos tambien el resultado del json
            nameFile =self.pathResponse+"responseTraining.json"
            with open(self.pathResponse+"responseTraining.json", 'w') as fp:
                json.dump(self.response, fp)

        elif self.algorithm == 5:#Gaussian

            self.response.update({"algorithm": "GaussianNB"})
            paramsData = {}
            self.response.update({"Params": "Default"})
            self.response.update({"Validation": "Cross Validation: " + str(self.validation)})

            errorData = {}
            try:

                #instancia al objeto...
                gaussianObject = GaussianNB.Gaussian(self.data, self.target, self.validation)
                if len(self.classArray)>2:
                    gaussianObject.trainingMethod(2)#multilabel
                else:
                    gaussianObject.trainingMethod(1)#binary

                performance = {}
                performance.update({"accuracy":gaussianObject.performanceData.scoreData[3]})
                performance.update({"recall": gaussianObject.performanceData.scoreData[4]})
                performance.update({"precision": gaussianObject.performanceData.scoreData[5]})
                performance.update({"f1": gaussianObject.performanceData.scoreData[6]})

                self.response.update({"Performance": performance})

                errorData.update({"exec_algorithm": "OK"})
            except:
                errorData.update({"exec_algorithm": "ERROR"})
                pass

            try:

                #learning curve
                learningCurveDemo = createLearningCurve.curveLearning(self.data, self.target, gaussianObject.model, self.validation, self.pathResponse)
                learningCurveDemo.createLearningCurve()
                errorData.update({"curveLearning" : "ok"})
            except:
                errorData.update({"curveLearning" : "error"})
                pass

            try:
                #confusion matrix data
                confusionMatrixDemo = createConfusionMatrix.confusionMatrix(self.data, self.target, gaussianObject.model, self.validation, self.pathResponse, self.classArray)
                responseMatrix = confusionMatrixDemo.createConfusionMatrix(self.dictTransform)
                self.response.update({"matrixConfusionResponse":responseMatrix})
                errorData.update({"confusionMatrix" : "ok"})
            except:
                errorData.update({"confusionMatrix" : "error"})
                pass

            self.response.update({"errorExec": errorData})

            #exportamos tambien el resultado del json
            nameFile =self.pathResponse+"responseTraining.json"
            with open(self.pathResponse+"responseTraining.json", 'w') as fp:
                json.dump(self.response, fp)

        elif self.algorithm == 6:#Gradient

            self.response.update({"algorithm": "GradientBoostingClassifier"})
            paramsData = {}
            paramsData.update({"n_estimators":self.params[0]})
            paramsData.update({"loss":self.params[1]})
            paramsData.update({"min_samples_leaf":self.params[2]})
            paramsData.update({"min_samples_split":self.params[3]})

            self.response.update({"Params": paramsData})
            self.response.update({"Validation": "Cross Validation: " + str(self.validation)})
            errorData = {}

            try:
                #instancia al objeto...
                gradientObject = Gradient.Gradient(self.data,self.target,int(self.params[0]), self.params[1], int(self.params[2]), int(self.params[3]),self.validation)
                if len(self.classArray)>2:
                    gradientObject.trainingMethod(2)#multilabel
                else:
                    gradientObject.trainingMethod(1)#binary

                performance = {}
                performance.update({"accuracy":gradientObject.performanceData.scoreData[3]})
                performance.update({"recall": gradientObject.performanceData.scoreData[4]})
                performance.update({"precision": gradientObject.performanceData.scoreData[5]})
                performance.update({"f1": gradientObject.performanceData.scoreData[6]})


                self.response.update({"Performance": performance})

                errorData.update({"exec_algorithm": "OK"})
            except:
                errorData.update({"exec_algorithm": "ERROR"})
                pass

            try:

                #learning curve
                learningCurveDemo = createLearningCurve.curveLearning(self.data, self.target, gradientObject.model, self.validation, self.pathResponse)
                learningCurveDemo.createLearningCurve()
                errorData.update({"curveLearning" : "ok"})
            except:
                errorData.update({"curveLearning" : "error"})
                pass

            try:
                #confusion matrix data
                confusionMatrixDemo = createConfusionMatrix.confusionMatrix(self.data, self.target, gradientObject.model, self.validation, self.pathResponse, self.classArray)
                responseMatrix = confusionMatrixDemo.createConfusionMatrix(self.dictTransform)
                self.response.update({"matrixConfusionResponse":responseMatrix})
                errorData.update({"confusionMatrix" : "ok"})
            except:
                errorData.update({"confusionMatrix" : "error"})
                pass

            self.response.update({"errorExec": errorData})

            #exportamos tambien el resultado del json
            nameFile =self.pathResponse+"responseTraining.json"
            with open(self.pathResponse+"responseTraining.json", 'w') as fp:
                json.dump(self.response, fp)

        elif self.algorithm == 7:#KNN

            self.response.update({"algorithm": "KNeighborsClassifier"})
            paramsData = {}
            paramsData.update({"n_neighbors":self.params[0]})
            paramsData.update({"algorithm":self.params[1]})
            paramsData.update({"metric":self.params[2]})
            paramsData.update({"weights":self.params[3]})

            self.response.update({"Params": paramsData})
            self.response.update({"Validation": "Cross Validation: " + str(self.validation)})
            errorData = {}

            try:
                #instancia al objeto...
                knnObect = knn.knn(self.data, self.target, int(self.params[0]), self.params[1], self.params[2], self.params[3], self.validation)
                if len(self.classArray)>2:
                    knnObect.trainingMethod(2)#multilabel
                else:
                    knnObect.trainingMethod(1)#binary

                performance = {}
                performance.update({"accuracy":knnObect.performanceData.scoreData[3]})
                performance.update({"recall": knnObect.performanceData.scoreData[4]})
                performance.update({"precision": knnObect.performanceData.scoreData[5]})
                performance.update({"f1": knnObect.performanceData.scoreData[6]})
                self.response.update({"Performance": performance})

                errorData.update({"exec_algorithm": "OK"})
            except:
                errorData.update({"exec_algorithm": "ERROR"})
                pass


            try:

                #learning curve
                learningCurveDemo = createLearningCurve.curveLearning(self.data, self.target, knnObect.model, self.validation, self.pathResponse)
                learningCurveDemo.createLearningCurve()
                errorData.update({"curveLearning" : "ok"})
            except:
                errorData.update({"curveLearning" : "error"})
                pass

            try:
                #confusion matrix data
                confusionMatrixDemo = createConfusionMatrix.confusionMatrix(self.data, self.target, knnObect.model, self.validation, self.pathResponse, self.classArray)
                responseMatrix = confusionMatrixDemo.createConfusionMatrix(self.dictTransform)
                self.response.update({"matrixConfusionResponse":responseMatrix})
                errorData.update({"confusionMatrix" : "ok"})
            except:
                errorData.update({"confusionMatrix" : "error"})
                pass

            self.response.update({"errorExec": errorData})

            #exportamos tambien el resultado del json
            nameFile =self.pathResponse+"responseTraining.json"
            with open(self.pathResponse+"responseTraining.json", 'w') as fp:
                json.dump(self.response, fp)

        elif self.algorithm == 8:#MLP

            self.response.update({"algorithm": "MLPClassifier"})
            paramsData = {}
            paramsData.update({"activation":self.params[0]})
            paramsData.update({"solver":self.params[1]})
            paramsData.update({"learning_rate":self.params[2]})
            paramsData.update({"hidden_layer_sizes_a":self.params[3]})
            paramsData.update({"hidden_layer_sizes_b":self.params[4]})
            paramsData.update({"hidden_layer_sizes_c":self.params[5]})
            paramsData.update({"alpha":self.params[6]})
            paramsData.update({"max_iter":self.params[7]})
            paramsData.update({"shuffle":self.params[8]})

            self.response.update({"Params": paramsData})
            self.response.update({"Validation": "Cross Validation: " + str(self.validation)})

            errorData = {}
            try:
                #instancia al objeto...
                MLPObject = MLP.MLP(self.data,self.target, self.params[0], self.params[1], self.params[2], int(self.params[3]), int(self.params[4]), int(self.params[5]), float(self.params[6]), int(self.params[7]), self.params[8], self.validation)
                if len(self.classArray)>2:
                    MLPObject.trainingMethod(2)#multilabel
                else:
                    MLPObject.trainingMethod(1)#binary

                performance = {}
                performance.update({"accuracy":MLPObject.performanceData.scoreData[3]})
                performance.update({"recall": MLPObject.performanceData.scoreData[4]})
                performance.update({"precision": MLPObject.performanceData.scoreData[5]})
                performance.update({"f1": MLPObject.performanceData.scoreData[6]})

                self.response.update({"Performance": performance})

                errorData.update({"exec_algorithm": "OK"})
            except:
                errorData.update({"exec_algorithm": "ERROR"})
                pass

            try:

                #learning curve
                learningCurveDemo = createLearningCurve.curveLearning(self.data, self.target, MLPObject.model, self.validation, self.pathResponse)
                learningCurveDemo.createLearningCurve()
                errorData.update({"curveLearning" : "ok"})
            except:
                errorData.update({"curveLearning" : "error"})
                pass

            try:
                #confusion matrix data
                confusionMatrixDemo = createConfusionMatrix.confusionMatrix(self.data, self.target, MLPObject.model, self.validation, self.pathResponse, self.classArray)
                responseMatrix = confusionMatrixDemo.createConfusionMatrix(self.dictTransform)
                self.response.update({"matrixConfusionResponse":responseMatrix})
                errorData.update({"confusionMatrix" : "ok"})
            except:
                errorData.update({"confusionMatrix" : "error"})
                pass

            self.response.update({"errorExec": errorData})

            #exportamos tambien el resultado del json
            nameFile =self.pathResponse+"responseTraining.json"
            with open(self.pathResponse+"responseTraining.json", 'w') as fp:
                json.dump(self.response, fp)

        elif self.algorithm == 9:#NuSVC

            self.response.update({"algorithm": "NuSVC"})
            paramsData = {}
            paramsData.update({"kernel":self.params[0]})
            paramsData.update({"nu":self.params[1]})
            paramsData.update({"degree":self.params[2]})
            paramsData.update({"gamma":self.params[3]})

            self.response.update({"Params": paramsData})
            self.response.update({"Validation": "Cross Validation: " + str(self.validation)})
            errorData = {}

            try:
                #instancia al objeto...
                nuSVM = NuSVM.NuSVM(self.data,self.target,self.params[0], float(self.params[1]), int(self.params[2]), float(self.params[3]), self.validation)
                if len(self.classArray)>2:
                    nuSVM.trainingMethod(2)#multilabel
                else:
                    nuSVM.trainingMethod(1)#binary

                performance = {}
                performance.update({"accuracy":nuSVM.performanceData.scoreData[3]})
                performance.update({"recall": nuSVM.performanceData.scoreData[4]})
                performance.update({"precision": nuSVM.performanceData.scoreData[5]})
                performance.update({"f1": nuSVM.performanceData.scoreData[6]})


                self.response.update({"Performance": performance})

                errorData.update({"exec_algorithm": "OK"})
            except:
                errorData.update({"exec_algorithm": "ERROR"})
                pass

            try:

                #learning curve
                learningCurveDemo = createLearningCurve.curveLearning(self.data, self.target, nuSVM.model, self.validation, self.pathResponse)
                learningCurveDemo.createLearningCurve()
                errorData.update({"curveLearning" : "ok"})
            except:
                errorData.update({"curveLearning" : "error"})
                pass

            try:
                #confusion matrix data
                confusionMatrixDemo = createConfusionMatrix.confusionMatrix(self.data, self.target, nuSVM.model, self.validation, self.pathResponse, self.classArray)
                responseMatrix = confusionMatrixDemo.createConfusionMatrix(self.dictTransform)
                self.response.update({"matrixConfusionResponse":responseMatrix})
                errorData.update({"confusionMatrix" : "ok"})
            except:
                errorData.update({"confusionMatrix" : "error"})
                pass

            self.response.update({"errorExec": errorData})

            #exportamos tambien el resultado del json
            nameFile =self.pathResponse+"responseTraining.json"
            with open(self.pathResponse+"responseTraining.json", 'w') as fp:
                json.dump(self.response, fp)

        elif self.algorithm == 10:#RandomForest

            self.response.update({"algorithm": "RandomForest"})
            paramsData = {}
            paramsData.update({"n_estimators":self.params[0]})
            paramsData.update({"criterion":self.params[1]})
            paramsData.update({"min_samples_split":self.params[2]})
            paramsData.update({"min_samples_leaf":self.params[3]})
            paramsData.update({"bootstrap":self.params[4]})

            self.response.update({"Params": paramsData})
            self.response.update({"Validation": "Cross Validation: " + str(self.validation)})
            errorData = {}

            try:
                #instancia al objeto...
                rf = RandomForest.RandomForest(self.data,self.target, int(self.params[0]),self.params[1], int(self.params[2]), int(self.params[3]), self.params[4], self.validation)
                if len(self.classArray)>2:
                    rf.trainingMethod(2)#multilabel
                else:
                    rf.trainingMethod(1)#binary

                performance = {}
                performance.update({"accuracy":rf.performanceData.scoreData[3]})
                performance.update({"recall": rf.performanceData.scoreData[4]})
                performance.update({"precision": rf.performanceData.scoreData[5]})
                performance.update({"f1": rf.performanceData.scoreData[6]})

                self.response.update({"Performance": performance})

                errorData.update({"exec_algorithm": "OK"})
            except:
                errorData.update({"exec_algorithm": "ERROR"})
                pass

            try:

                #learning curve
                learningCurveDemo = createLearningCurve.curveLearning(self.data, self.target, rf.model, self.validation, self.pathResponse)
                learningCurveDemo.createLearningCurve()
                errorData.update({"curveLearning" : "ok"})
            except:
                errorData.update({"curveLearning" : "error"})
                pass

            try:
                #confusion matrix data
                confusionMatrixDemo = createConfusionMatrix.confusionMatrix(self.data, self.target, rf.model, self.validation, self.pathResponse, self.classArray)
                responseMatrix = confusionMatrixDemo.createConfusionMatrix(self.dictTransform)
                self.response.update({"matrixConfusionResponse":responseMatrix})
                errorData.update({"confusionMatrix" : "ok"})
            except:
                errorData.update({"confusionMatrix" : "error"})
                pass

            self.response.update({"errorExec": errorData})

            #exportamos tambien el resultado del json
            nameFile =self.pathResponse+"responseTraining.json"
            with open(self.pathResponse+"responseTraining.json", 'w') as fp:
                json.dump(self.response, fp)

        else:#SVC

            self.response.update({"algorithm": "SVC"})
            paramsData = {}
            paramsData.update({"kernel":self.params[0]})
            paramsData.update({"C_value":self.params[1]})
            paramsData.update({"degree":self.params[2]})
            paramsData.update({"gamma":self.params[3]})

            self.response.update({"Params": paramsData})
            self.response.update({"Validation": "Cross Validation: " + str(self.validation)})
            errorData = {}

            try:
                #instancia al objeto...
                svm = SVM.SVM(self.data, self.target, self.params[0], float(self.params[1]), int(self.params[2]), float(self.params[3]), self.validation)
                if len(self.classArray)>2:
                    svm.trainingMethod(2)#multilabel
                else:
                    svm.trainingMethod(1)#binary

                performance = {}
                performance.update({"accuracy":svm.performanceData.scoreData[3]})
                performance.update({"recall": svm.performanceData.scoreData[4]})
                performance.update({"precision": svm.performanceData.scoreData[5]})
                performance.update({"f1": svm.performanceData.scoreData[6]})
                self.response.update({"Performance": performance})

                errorData.update({"exec_algorithm": "OK"})
            except:
                errorData.update({"exec_algorithm": "ERROR"})
                pass

            try:

                #learning curve
                learningCurveDemo = createLearningCurve.curveLearning(self.data, self.target, svm.model, self.validation, self.pathResponse)
                learningCurveDemo.createLearningCurve()
                errorData.update({"curveLearning" : "ok"})
            except:
                errorData.update({"curveLearning" : "error"})
                pass

            try:
                #confusion matrix data
                confusionMatrixDemo = createConfusionMatrix.confusionMatrix(self.data, self.target, svm.model, self.validation, self.pathResponse, self.classArray)
                responseMatrix = confusionMatrixDemo.createConfusionMatrix(self.dictTransform)
                self.response.update({"matrixConfusionResponse":responseMatrix})
                errorData.update({"confusionMatrix" : "ok"})
            except:
                errorData.update({"confusionMatrix" : "error"})
                pass

            self.response.update({"errorExec": errorData})

            #exportamos tambien el resultado del json
            nameFile =self.pathResponse+"responseTraining.json"
            with open(self.pathResponse+"responseTraining.json", 'w') as fp:
                json.dump(self.response, fp)
