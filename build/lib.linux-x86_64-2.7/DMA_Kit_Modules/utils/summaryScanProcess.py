########################################################################
# summaryScanProcess.py,
#
# Create a summary of scan process 
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
from DMA_Kit_Modules.graphic import createCharts

class summaryProcessClusteringScan(object):

    def __init__(self, dataSetFile, pathResponse, performance):

        self.dataSetFile = dataSetFile
        self.dataFrame = pd.read_csv(dataSetFile)
        self.pathResponse = pathResponse
        self.performance = performance

    #metodo que permite hacer los histogramas por las medidas de desempeno
    def createHistogram(self):

        graphic = createCharts.graphicsCreator()

        for key in self.performance:
            print "Create histogram for ", key
            namePicture = self.pathResponse+key+".svg"
            title = "Histogram for "+key
            graphic.generateHistogram(self.dataFrame, key, namePicture, title)

    #metodo que permite crear el resumen estadistico
    def createStatisticSummary(self):

        matrixResponse = []
        header = ["Performance", "Mean", "STD", "Variance", "Min", "Max"]

        #trabajamos con las estadisticas...
        for key in self.performance:
            try:
                print "Process ", key
                row = []
                row.append(key)
                row.append(np.mean(self.dataFrame[key]))
                row.append(np.std(self.dataFrame[key]))
                row.append(np.var(self.dataFrame[key]))
                row.append(min(self.dataFrame[key]))
                row.append(max(self.dataFrame[key]))
                matrixResponse.append(row)
            except:
                pass

        df = pd.DataFrame(matrixResponse, columns=header)
        df.to_csv(self.pathResponse+"summaryStatistical.csv", index=False)

    #metodo que permite poder ordenar los arreglos de mayor a menor segun su valor
    def createRankingFile(self):

        #por cada medida, generamos un ranking con los primeros 10 modelos resultantes
        for data in self.performance:

            rankingPerformance = self.dataFrame.sort_values(data,ascending=False)
            rankingPerformance.to_csv(self.pathResponse+data+"_ranking.csv", index=False)
