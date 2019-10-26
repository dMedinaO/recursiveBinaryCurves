########################################################################
# DecisionTree.py,
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

class DecisionTree(object):

    def __init__ (self, dataset, response, criterion, splitter):
        self.dataset=dataset
        self.response=response
        self.criterion=criterion
        self.splitter=splitter

    def trainingMethod(self):
        self.model=tree.DecisionTreeRegressor(criterion=self.criterion,splitter=self.splitter)
        self.DecisionTreeAlgorithm=self.model.fit(self.dataset,self.response)
        self.predicctions = self.DecisionTreeAlgorithm.predict(self.dataset)
        self.r_score = self.DecisionTreeAlgorithm.score(self.dataset, self.response)
