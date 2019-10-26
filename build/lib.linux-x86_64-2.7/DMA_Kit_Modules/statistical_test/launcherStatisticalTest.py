########################################################################
# launcherStatisticalTest.py,
#
# Receives a dataset and calculates statistical tests such as:
# Pearson Coeficient
# Spearman Coeficient
# Kendalltau Coeficient
# Mann Whitney test
# Kolmogorov Smirnoff test
# Shapiro-Wilk Test
# T-Test
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

from modulesProject.statistical_test import getResultsFromTest
import pandas as pd
import json

#definicion de la clase
class statisticalTest(object):

    def __init__(self, dataSet, optionProcess):

        self.dataSet = dataSet
        self.optionProcess = int(optionProcess)
        self.data = pd.read_csv(dataSet)
        self.listKey = []
        for element in self.data:
            self.listKey.append(element)
        self.response = {}

    def checkExec(self):
        if self.optionProcess == 1:#Coeficiente de pearson
            print "Pearson Coeficient"
            try:
                dataObject1 = getResultsFromTest.ResultsFromTest()
                ResultFromPearson=dataObject1.Pearson(self.data[self.listKey[0]], self.data[self.listKey[1]])
                self.response.update({"statistical_test": "Pearson"})
                self.response.update({"statistic_value": ResultFromPearson[0]})
                self.response.update({"p_value": ResultFromPearson[1]})
                self.response.update({"exec": "OK"})
            except:
                self.response.update({"statistical_test": "Pearson"})
                self.response.update({"exec": "ERROR"})
                pass

        elif self.optionProcess == 2:#Coeficiente de spearman
            print "Spearman Coeficient"
            try:
                dataObject2 = getResultsFromTest.ResultsFromTest()
                ResultFromSpearman=dataObject2.Spearman(self.data[self.listKey[0]], self.data[self.listKey[1]])
                self.response.update({"statistical_test": "Spearman"})
                self.response.update({"statistic_value": ResultFromSpearman[0]})
                self.response.update({"p_value": ResultFromSpearman[1]})
                self.response.update({"exec": "OK"})
            except:
                self.response.update({"statistical_test": "Spearman"})
                self.response.update({"exec": "ERROR"})
                pass

        elif self.optionProcess == 3:#Coeficiente kendalltau
            print "Kendalltau Coeficient"
            try:
                dataObject3 = getResultsFromTest.ResultsFromTest()
                ResultFromKendalltau=dataObject3.Kendalltau(self.data[self.listKey[0]], self.data[self.listKey[1]])
                self.response.update({"statistical_test": "Kendalltau"})
                self.response.update({"statistic_value": ResultFromKendalltau[0]})
                self.response.update({"p_value": ResultFromKendalltau[1]})
                self.response.update({"exec": "OK"})
            except:
                self.response.update({"statistical_test": "Kendalltau"})
                self.response.update({"exec": "ERROR"})
                pass

        elif self.optionProcess == 4:#Mann Whitney
            print "Mann Whitney test"
            try:
                dataObject4 = getResultsFromTest.ResultsFromTest()
                ResultFromMannWhitney=dataObject4.MannWhitney(self.data[self.listKey[0]], self.data[self.listKey[1]])
                self.response.update({"statistical_test": "MannWhitney"})
                self.response.update({"statistic_value": ResultFromMannWhitney[0]})
                self.response.update({"p_value": ResultFromMannWhitney[1]})
                self.response.update({"exec": "OK"})
            except:
                self.response.update({"statistical_test": "MannWhitney"})
                self.response.update({"exec": "ERROR"})
                pass

        elif self.optionProcess == 5:#Kolmogorov
            print "Kolmogorov Smirnoff test"
            try:
                dataObject5 = getResultsFromTest.ResultsFromTest()
                ResultFromKolmogorov=dataObject5.Kolmogorov(self.data[self.listKey[0]])
                self.response.update({"statistical_test": "Kolmogorov-Smirnoff"})
                self.response.update({"statistic_value": ResultFromKolmogorov[0]})
                self.response.update({"p_value": ResultFromKolmogorov[1]})
                self.response.update({"exec": "OK"})
            except:
                self.response.update({"statistical_test": "Kolmogorov-Smirnoff"})
                self.response.update({"exec": "ERROR"})
                pass

        elif self.optionProcess == 6:#Shapiro
            print "Shapiro-Wilk Test"
            try:
                dataObject6 = getResultsFromTest.ResultsFromTest()
                ResultFromShapiro=dataObject6.Shapiro(self.data[self.listKey[0]])
                self.response.update({"statistical_test": "Shapiro-Wilk"})
                self.response.update({"statistic_value": ResultFromShapiro[0]})
                self.response.update({"p_value": ResultFromShapiro[1]})
                self.response.update({"exec": "OK"})
            except:
                self.response.update({"statistical_test": "Shapiro-Wilk"})
                self.response.update({"exec": "ERROR"})
                pass

        else:#T test
            print "T-Test"
            try:
                dataObject8 = getResultsFromTest.ResultsFromTest()
                ResultFromT_test=dataObject8.T_test(self.data[self.listKey[0]], self.data[self.listKey[1]])
                self.response.update({"statistical_test": "T-Test"})
                self.response.update({"statistic_value": ResultFromT_test[0]})
                self.response.update({"p_value": ResultFromT_test[1]})
                self.response.update({"exec": "OK"})
            except:
                self.response.update({"statistical_test": "T-Test"})
                self.response.update({"exec": "ERROR"})
                pass
