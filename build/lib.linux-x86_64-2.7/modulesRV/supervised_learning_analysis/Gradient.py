########################################################################
# Gradient.py,
#
# Executes Gradient model for a given dataset
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

from sklearn.ensemble import GradientBoostingClassifier
import responseTraining

class Gradient(object):

    def __init__ (self,dataset,target,n_estimators, loss, min_samples_split, min_samples_leaf, validation):
        self.dataset=dataset
        self.target=target
        self.n_estimators=n_estimators
        self.loss = loss
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.validation=validation

    def trainingMethod(self, kindDataSet):

        self.model= GradientBoostingClassifier(n_estimators=self.n_estimators)
        self.GradientAlgorithm= self.model.fit(self.dataset,self.target)

        params = "n_estimators:%d-loss:%s-min_samples_leaf:%d-min_samples_split:%d" % (self.n_estimators, self.loss, self.min_samples_leaf, self.min_samples_split)
        self.performanceData = responseTraining.responseTraining(self.GradientAlgorithm, 'Gradient', params, self.validation)

        if kindDataSet == 1:
            self.performanceData.estimatedMetricsPerformance(self.dataset, self.target)
        else:
            self.performanceData.estimatedMetricsPerformanceMultilabels(self.dataset, self.target)
