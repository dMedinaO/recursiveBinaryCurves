########################################################################
# setup.py,
#
# Executes SVM model for a given dataset
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

from sklearn import svm
import responseTraining

class SVM(object):

    #building
    def __init__(self,dataset, target, kernel, C_value, degree, gamma, validation):

        #init attributes values...
        self.dataset=dataset
        self.target=target
        self.kernel=kernel
        self.validation=validation
        self.C_value = C_value
        self.degree = degree
        self.gamma = gamma

    #instance training...
    def trainingMethod(self, kindDataSet):

        self.model=svm.SVC(kernel=self.kernel, degree=self.degree, gamma=self.gamma, C=self.C_value, probability=True)
        self.SVMAlgorithm =self.model.fit(self.dataset,self.target)

        params = "kernel:%s-C:%f-degree:%f-gamma:%f" % (self.kernel, self.C_value, self.degree, self.gamma)
        self.performanceData = responseTraining.responseTraining(self.SVMAlgorithm, 'SVM', params, self.validation)

        if kindDataSet == 1:
            self.performanceData.estimatedMetricsPerformance(self.dataset, self.target)
        else:
            self.performanceData.estimatedMetricsPerformanceMultilabels(self.dataset, self.target)
