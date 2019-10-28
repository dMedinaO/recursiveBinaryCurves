########################################################################
# MLP.py,
#
# Executes MLP model for a given dataset
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

from sklearn.neural_network import MLPClassifier
import responseTraining

class MLP(object):

    def __init__ (self,dataset,target,activation, solver, learning_rate, hidden_layer_sizes_a,hidden_layer_sizes_b,hidden_layer_sizes_c, alpha, max_iter, shuffle, validation):
        self.dataset=dataset
        self.target=target
        self.activation=activation
        self.solver=solver
        self.learning_rate=learning_rate
        self.hidden_layer_sizes=[hidden_layer_sizes_a,hidden_layer_sizes_b,hidden_layer_sizes_c]
        self.alpha = alpha
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.validation=validation

    def trainingMethod(self, kindDataSet):

        self.model=MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes,activation=self.activation,solver=self.solver,learning_rate=self.learning_rate)
        self.MLPAlgorithm=self.model.fit(self.dataset,self.target)

        params = "activation:%s-learning_rate:%s-solver:%s-hidden_layer_sizes_a:%d-hidden_layer_sizes_b:%d-hidden_layer_sizes_c:%d-alpha:%f-max_iter:%d-shuffle:%s" % (self.activation, self.learning_rate, self.solver,self.hidden_layer_sizes[0], self.hidden_layer_sizes[1], self.hidden_layer_sizes[2], self.alpha, self.max_iter, self.shuffle)
        self.performanceData = responseTraining.responseTraining(self.MLPAlgorithm, 'MLP', params, self.validation)

        if kindDataSet == 1:
            self.performanceData.estimatedMetricsPerformance(self.dataset, self.target)
        else:
            self.performanceData.estimatedMetricsPerformanceMultilabels(self.dataset, self.target)
