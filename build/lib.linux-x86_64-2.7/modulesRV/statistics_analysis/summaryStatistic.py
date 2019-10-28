########################################################################
# summaryStatistic.py,
#
# Estimates statistician for a dataset.
# Makes a summary of feature.
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
#######################################################################

import numpy as np
import pandas as pd

class createStatisticSummary(object):

    def __init__(self, dataSet):

        self.dataSet = pd.read_csv(dataSet)

    #metodo que permite calcular los estadisticos para una columna en el set de datos...
    def calculateValuesForColumn(self, attribute):

        dictResponse = {}
        dictResponse.update({'mean': np.mean(self.dataSet[attribute])})
        dictResponse.update({'std': np.std(self.dataSet[attribute])})
        dictResponse.update({'var': np.var(self.dataSet[attribute])})
        dictResponse.update({'max': max(self.dataSet[attribute])})
        dictResponse.update({'min': min(self.dataSet[attribute])})

        return dictResponse
