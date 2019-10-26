########################################################################
# processStatisticsSummary.py,
#
# Receives continuous type data, then do statisticals calculations
# Information summary
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

from modulesProject.statistics_analysis import getFeatures
import numpy as np

class statisticsValues(object):

    def __init__(self, headerFeatures):

        self.headerFeatures = headerFeatures

    #metodo que permite obtener los datos continuos del set de datos
    def getValuesContinues(self):

        matrixResponse = []

        for feature in self.headerFeatures.listFeatures:
            if feature.kindData == "CONTINUA":
                matrixResponse.append(self.getStatistical(feature.data, feature.nameData))

        return matrixResponse

    #metodo que permite hacer los calculos asociados a las estadisticas de un atributo
    def getStatistical(self, data, nameData):

        mean = np.mean(data)
        std = np.std(data)
        variance = np.var(data)
        maxValue = max(data)
        minValue = min(data)

        row = [nameData, mean, std, variance, maxValue, minValue]

        return row
