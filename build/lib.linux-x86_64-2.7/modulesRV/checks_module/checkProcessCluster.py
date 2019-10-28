########################################################################
# checkProcessCluster.py,
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

import pandas as pd
import numpy as np
import math
from scipy import special

class checkProcess(object):

    def __init__(self, dataFrame):

        self.dataFrame = dataFrame

        #obtenemos los maximos coeficientes
        maxCalinski = max(self.dataFrame['calinski_harabaz_score'])
        maxSiluetas = max(self.dataFrame['silhouette_score'])

        print maxCalinski
        print maxSiluetas
        self.candidato = self.getCandidateIndexScore(maxCalinski, maxSiluetas)

    #funcion que permite obtener los candidatos con los valores maximos de calinski y siluetas
    def getCandidateIndexScore(self, maxCalinski, maxSiluetas):

        indexCandCal = []
        indexCandSil = []

        index=0
        for element in self.dataFrame['calinski_harabaz_score']:
            if element == maxCalinski:
                indexCandCal.append(index)
            index+=1

        index=0
        for element in self.dataFrame['silhouette_score']:
            if element == maxSiluetas:
                indexCandSil.append(index)
            index+=1

        print indexCandCal
        print indexCandSil

        #buscamos los calisnki que estan en ambas listas, si no existen, solo tomamos el primer elemento
        indexCandidato = -1

        for element in indexCandCal:
            if element in indexCandSil:
                indexCandidato=element
                break

        if indexCandidato == -1:
            indexCandidato = indexCandCal[0]

        return indexCandidato

    #funcion que permite evaluar la cantidad de ejemplos por division
    def checkSplitter(self, member1, member2, threshold, sizeEval):
        total = sizeEval

        member1= float(member1)/float(total)*100
        member2= float(member2)/float(total)*100
        print "member1: ", member1
        print "member2: ", member2
        if member1<threshold or member2< threshold:#no cumple con criterio de tamano
            return -1
        else:#si cumple con criterio de tamano
            return 1

    #funcion que permite evaluar la proporcion de las clases si corresponde
    def checkEvalClass(self, listResponse, threshold):

        classElement = list(set(listResponse))

        if len(classElement)>1:

            arrayProportion = []

            for element in classElement:
                count=0
                for data in listResponse:
                    if data == element:
                        count+=1
                count = float(count)/len(listResponse) * 100#sacamos el porcentaje

                arrayProportion.append(count)

            response=0
            print arrayProportion
            #evaluamos si existe desbalance
            for proportion in arrayProportion:
                if proportion <= threshold:
                    response=-1
                    break

            if response == 0:
                return 0#el conjunto de datos se encuentra balanceado
            else:
                return -1
        else:
            return -1

    #metodo que permite evaluar de manera estadistica los cluster
    def checkStatisticValidation(self, matrixDataG1, matrixDataG2, significancia):

        #obtenemos los valores estadisticos
        meanG1, stdG1 = self.getStatisticValues(matrixDataG1)
        meanG2, stdG2 = self.getStatisticValues(matrixDataG2)

        #hacemos la aplicacion de los test
        responseP1 = self.comparePointToPoint(meanG1, meanG2, stdG1, stdG2)
        responseP2 = self.compareUnderCurve(meanG1, meanG2, stdG1, stdG2)
        responseP3 = self.compareUnderCurve(meanG1, meanG2, stdG1, stdG2)

        #aplicamos test de siems
        listPvalue = [responseP1, responseP2, responseP3]
        print listPvalue
        listPvalue.sort()#ordenamos los valores
        print listPvalue

        #obtenemos el valor real
        pValueCombined = []
        for i in range(3):
            value = 1/float(i+1)*listPvalue[i]
            pValueCombined.append(value)

        print pValueCombined
        print min(pValueCombined)

        if min(pValueCombined)<=significancia:
            print "rechazo H0"
            return 1#rechazo H0
        else:
            print "no rechazo H0"
            return 0#no rechazo H0

    #metodo que permite aplicar el test de comparacion punto-punto
    def comparePointToPoint(self, meanG1, meanG2, stdG1, stdG2):

        statistic = 0

        for i in range(len(meanG1)):
            num = (float(meanG1[i]) - float(meanG2[i]))*(float(meanG1[i]) - float(meanG2[i]))
            den = float(stdG1[i])**2 + float(stdG2[i])**2
            statistic+= float(num)/float(den)

        #obtenemos la CDF
        gradosLibertad = len(meanG1)
        cdf = 1/(math.gamma(float(gradosLibertad)/2.0))*(special.gammainc(float(gradosLibertad)/2.0, float(statistic)/2.0))
        pvalue = 1-cdf

        return pvalue

    #metodo que permite aplicar el test de comparacion de area bajo la curva
    def compareUnderCurve(self, meanG1, meanG2, stdG1, stdG2):

        statistic = 0

        for i in range(len(meanG1)-1):
            num = (float(meanG1[i]) - float(meanG2[i]) + float(meanG1[i+1]) - float(meanG2[i+1]))**2
            den = float(stdG1[i])**2 + float(stdG2[i])**2 + float(stdG1[i+1])**2 + float(stdG2[i+1])**2
            statistic+= float(num)/float(den)

        #obtenemos la CDF
        gradosLibertad = len(meanG1)-1
        cdf = 1/(math.gamma(float(gradosLibertad)/2.0))*(special.gammainc(float(gradosLibertad)/2.0, float(statistic)/2.0))
        pvalue = 1-cdf

        return pvalue

    #metodo que permite aplicar el test de comparacion de area bajo la curva
    def compareSlopesCurve(self, meanG1, meanG2, stdG1, stdG2):

        statistic = 0

        for i in range(len(meanG1)-1):
            num = (float(meanG2[i]) - float(meanG1[i]) + float(meanG2[i+1]) - float(meanG1[i+1]))**2
            den = float(stdG1[i])**2 + float(stdG2[i])**2 + float(stdG1[i+1])**2 + float(stdG2[i+1])**2
            statistic+= float(num)/float(den)

        #obtenemos la CDF
        gradosLibertad = len(meanG1)-1
        cdf = 1/(math.gamma(float(gradosLibertad)/2.0))*(special.gammainc(float(gradosLibertad)/2.0, float(statistic)/2.0))
        pvalue = 1-cdf

        return pvalue

    #metodo que permite poder obtener las estadisticas basicas (promedio y desviacion estandar) de la data
    def getStatisticValues(self, matrixData):

        arrayMean = []
        arraySTD = []

        for i in range(len(matrixData[0])):
            columnData = []
            for j in range(len(matrixData)):
                columnData.append(matrixData[j][i])
            arrayMean.append(np.mean(columnData))
            arraySTD.append(np.std(columnData))

        return arrayMean, arraySTD
