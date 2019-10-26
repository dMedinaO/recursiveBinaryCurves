########################################################################
# performanceData.py,
#
# Calculates perfomance metrics for generated outputs
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

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import math

class performancePrediction(object):

    def __init__(self, realValues, predictValues):

        self.realValues = realValues
        self.predictValues = predictValues

    #metodo que permite calcular el coeficiente de pearson...
    def calculatedPearson(self):

        response = pearsonr(self.realValues, self.predictValues)
        if math.isnan(response[0]):
            r1 = 'ERROR'
        else:
            r1 = response[0]

        if math.isnan(response[1]):
            r2 = 'ERROR'
        else:
            r2 = response[1]

        dictResponse = {"pearsonr": r1, "pvalue": r2}
        return dictResponse

    #metodo que permite calcular el coeficiente de spearman...
    def calculatedSpearman(self):

        response = spearmanr(self.realValues, self.predictValues)

        if math.isnan(response[0]):
            r1 = 'ERROR'
        else:
            r1 = response[0]

        if math.isnan(response[1]):
            r2 = 'ERROR'
        else:
            r2 = response[1]

        dictResponse = {"spearmanr": r1, "pvalue": r2}
        return dictResponse

    #metodo que permite calcular el kendalltau...
    def calculatekendalltau(self):

        response = kendalltau(self.realValues, self.predictValues)

        if math.isnan(response[0]):
            r1 = 'ERROR'
        else:
            r1 = response[0]

        if math.isnan(response[1]):
            r2 = 'ERROR'
        else:
            r2 = response[1]

        dictResponse = {"kendalltau": r1, "pvalue": r2}
        return dictResponse
