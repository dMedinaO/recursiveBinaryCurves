########################################################################
# knn.py,
#
# Executes KNN model for a given dataset
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

#modules import
from sklearn.neighbors import KNeighborsClassifier
import responseTraining

class knn(object):

    #building class...
    def __init__(self, dataset, response, n_neighbors, algorithm, metric,  weights, validation):

        #init attributes values...
        self.dataset = dataset
        self.response = response
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.weights = weights
        self.validation = validation

    #instance training...
    def trainingMethod(self, kindDataSet):

        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights, algorithm=self.algorithm, metric=self.metric, n_jobs=-1)#instancia
        self.knnAlgorithm = self.model.fit(self.dataset, self.response)

        #training...
        params = "algorithm:%s-metric:%s-neighbors:%d-weights:%s" % (self.algorithm, self.metric, self.n_neighbors, self.weights)
        self.performanceData = responseTraining.responseTraining(self.knnAlgorithm, 'KNN', params, self.validation)

        if kindDataSet == 1:
            self.performanceData.estimatedMetricsPerformance(self.dataset, self.response)
        else:
            self.performanceData.estimatedMetricsPerformanceMultilabels(self.dataset, self.response)
