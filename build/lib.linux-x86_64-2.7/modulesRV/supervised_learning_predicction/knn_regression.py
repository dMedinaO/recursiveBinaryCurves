########################################################################
# knn_regression.py,
#
# Executes knn model for a given dataset
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
from sklearn.neighbors import KNeighborsRegressor

class KNN_Model(object):

    #building class...
    def __init__(self, dataset, response, n_neighbors, algorithm, metric,  weights):

        #init attributes values...
        self.dataset = dataset
        self.response = response
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.weights = weights

    #instance training...
    def trainingMethod(self):

        self.model = KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=self.weights, algorithm=self.algorithm, metric=self.metric, n_jobs=-1)#instancia
        self.KNN_model =self.model.fit(self.dataset,self.response)
        self.predicctions = self.KNN_model.predict(self.dataset)
        self.r_score = self.KNN_model.score(self.dataset, self.response)
