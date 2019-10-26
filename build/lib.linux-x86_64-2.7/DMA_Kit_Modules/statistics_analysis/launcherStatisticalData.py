########################################################################
# launcherStatisticalData.py,
#
# Execute statistical analysis.
# Receives a dataset,a column key of it, and the filename for the output
# Generates outputs files and change process state
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

from DMA_Kit_Modules.statistics_analysis import processStatiticsSummary
from DMA_Kit_Modules.graphic import createCharts

import json
import pandas as pd
import numpy as np

#definicion de la clase
class launcherStatisticalProcess(object):

    def __init__(self, dataSet, pathResponse, optionProcess, keyFeature):

        self.dataSet = dataSet
        self.pathResponse = pathResponse
        self.optionProcess = int(optionProcess)#el metodo a ejecutar...
        self.keyFeature = keyFeature#key del data set, solo en los casos que corresponde

        #removemos las variables categoricas distintas al key...
        columnsRemove = []
        for key in self.dataSet:
            response=0
            for i in range(len(self.dataSet)):
                try:
                    values = int(self.dataSet[key][i])
                except:
                    response=1
                    columnsRemove.append(key)
                    break
        for key in columnsRemove:
            if key != self.keyFeature:
                self.dataSet.drop([key], axis='columns', inplace=True)

    #metodo que permite evaluar la opcion a ejecutar...
    def checkExec(self):

        if self.optionProcess == 1:#show data continue

            try:
                graphic = createCharts.graphicsCreator()
                namePicture = self.pathResponse+"viewContinueValuesFor_"+self.keyFeature+".png"
                graphic.createScatterContinueData(self.dataSet[self.keyFeature], namePicture, self.keyFeature)
                print "Create graphic OK"
            except:
                print "Error during create graphic"
                pass

        elif self.optionProcess == 2:#boxplot and violinplot

            try:
                graphic = createCharts.graphicsCreator()
                namePicture = self.pathResponse+"boxplot.svg"
                graphic.createBoxPlot(self.dataSet, namePicture)
                print "Box plot graphic OK"
            except:
                print "Error during create BoxPlot"
                pass

            try:
                graphic = createCharts.graphicsCreator()
                namePicture = self.pathResponse+"violinplot.svg"
                graphic.createViolinPlot(self.dataSet, namePicture)
                print "Violin plot graphic OK"
            except:
                print "Error during create Violin"
                pass

        elif self.optionProcess == 3:#histograma

            try:
                graphic = createCharts.graphicsCreator()
                namePicture = self.pathResponse+"histogram_"+self.keyFeature+".svg"
                title = "Histogram for feature "+self.keyFeature
                graphic.generateHistogram(self.dataSet, self.keyFeature, namePicture, title)
                print "create histogram for feature: ", self.keyFeature
            except:
                print "Error during create Histogram"
                pass

        elif self.optionProcess == 4:#frequence

            try:
                keys = list(set(self.dataSet[self.keyFeature]))
                values = []
                for key in keys:
                    cont=0
                    for i in range(len(self.dataSet[self.keyFeature])):
                        if self.dataSet[self.keyFeature][i] == key:
                            cont+=1
                    values.append(cont)
                namePicture = self.pathResponse+"piechartFor_"+self.keyFeature+".svg"
                graphic = createCharts.graphicsCreator()
                graphic.createPieChart(keys, values, namePicture)
                print "Create pie chart for "+self.keyFeature
            except:
                print "Error during create a pie chart"
                pass

        elif self.optionProcess == 5:#parallel
            try:
                graphic = createCharts.graphicsCreator()
                namePicture = self.pathResponse+"parallel_coordinates_"+self.keyFeature+".svg"
                title = "parallel_coordinates for "+self.keyFeature
                graphic.createParallelCoordinates(self.dataSet, self.keyFeature, namePicture, title)
                print "Create parallel_coordinates graphic"
            except:
                print "Error during create a parallel_coordinates"
                pass

        elif self.optionProcess == 6:#SPLOM

            try:
                graphic = createCharts.graphicsCreator()
                namePicture = self.pathResponse+"splom.svg"
                graphic.createScatterPlotMatrix(self.dataSet, namePicture, self.keyFeature)
                print "Create SPLOM for feature ", self.keyFeature
            except:
                print "Error during create SPLOM"
                pass

        else:

            matrixResponse = []
            header = ["Feature", "Mean", "STD", "Variance", "Min", "Max"]

            #trabajamos con las estadisticas...
            for key in self.dataSet:
                try:
                    print "Process ", key
                    row = []
                    row.append(key)
                    row.append(np.mean(self.dataSet[key]))
                    row.append(np.std(self.dataSet[key]))
                    row.append(np.var(self.dataSet[key]))
                    row.append(min(self.dataSet[key]))
                    row.append(max(self.dataSet[key]))
                    matrixResponse.append(row)
                except:
                    pass

            df = pd.DataFrame(matrixResponse, columns=header)
            df.to_csv(self.pathResponse+"summaryStatistical.csv", index=False)

            print "Create summaryStatistical.csv file"
