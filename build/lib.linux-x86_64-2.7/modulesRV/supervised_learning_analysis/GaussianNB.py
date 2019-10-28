########################################################################
# GaussianNB.py,
#
# Executes gaussian model for a given dataset
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

from sklearn.naive_bayes import GaussianNB
import responseTraining

class Gaussian(object):

    def __init__(self,dataset,target,validation):
        self.dataset=dataset
        self.target=target
        self.validation=validation

    def trainingMethod(self, kindDataSet):

        self.model=GaussianNB()
        self.GaussianNBAlgorithm=self.model.fit(self.dataset,self.target)

        if kindDataSet == 1:
            params = 'Params:default'
            self.performanceData = responseTraining.responseTraining(self.GaussianNBAlgorithm, 'GaussianNB', params, self.validation)
            self.performanceData.estimatedMetricsPerformance(self.dataset, self.target)
        else:
            params = 'Params:default'
            self.performanceData = responseTraining.responseTraining(self.GaussianNBAlgorithm, 'GaussianNB', params, self.validation)
            self.performanceData.estimatedMetricsPerformanceMultilabels(self.dataset, self.target)
