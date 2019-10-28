########################################################################
# setup.py,
#
# Executes desicion tree model for a given dataset
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

from sklearn import tree
import responseTraining

class DecisionTree(object):

    def __init__ (self, dataset, target, criterion, splitter,validation):
        self.dataset=dataset
        self.target=target
        self.criterion=criterion
        self.splitter=splitter
        self.validation=validation

    def trainingMethod(self, kindDataSet):
        self.model=tree.DecisionTreeClassifier(criterion=self.criterion,splitter=self.splitter)
        self.DecisionTreeAlgorithm=self.model.fit(self.dataset,self.target)

        if kindDataSet == 1:
            params = "criterion:%s-splitter:%s" % (self.criterion,self.splitter)
            self.performanceData = responseTraining.responseTraining(self.DecisionTreeAlgorithm, 'DecisionTree', params, self.validation)
            self.performanceData.estimatedMetricsPerformance(self.dataset, self.target)

        else:
            params = "criterion:%s-splitter:%s" % (self.criterion,self.splitter)
            self.performanceData = responseTraining.responseTraining(self.DecisionTreeAlgorithm, 'DecisionTree', params, self.validation)
            self.performanceData.estimatedMetricsPerformanceMultilabels(self.dataset, self.target)
