########################################################################
# evaluationClustering.py,
#
#Evaluate a clustering algorithm's results with theis execution parameters.
#
#Receives dataset and clustering's results labels.
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

from sklearn import metrics

class evaluationClustering(object):

    def __init__(self, dataSet, labelsResponse):

        self.dataSet = dataSet
        self.labelsResponse = labelsResponse
        try:
            self.calinski = metrics.calinski_harabaz_score(self.dataSet, self.labelsResponse)
            self.siluetas = metrics.silhouette_score(self.dataSet, self.labelsResponse, metric='euclidean')
        except:
            self.calinski = 0#tirar una moneda
            self.siluetas = 0#tirar una moneda
            pass
