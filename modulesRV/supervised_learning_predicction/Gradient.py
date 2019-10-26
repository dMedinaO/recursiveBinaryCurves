########################################################################
# Gradient.py,
#
# Executes Gradient model for a given dataset.
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

from sklearn.ensemble import GradientBoostingRegressor

class Gradient(object):

    def __init__ (self,dataset,response,n_estimators, loss, criterion, min_samples_split, min_samples_leaf):
        self.dataset=dataset
        self.response=response
        self.n_estimators=n_estimators
        self.loss = loss
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split

    def trainingMethod(self):

        self.model= GradientBoostingRegressor(n_estimators=self.n_estimators, loss=self.loss, criterion=self.criterion, min_samples_leaf=self.min_samples_leaf, min_samples_split=self.min_samples_split)
        self.GradientAlgorithm= self.model.fit(self.dataset,self.response)
        self.predicctions = self.GradientAlgorithm.predict(self.dataset)
        self.r_score = self.GradientAlgorithm.score(self.dataset, self.response)
