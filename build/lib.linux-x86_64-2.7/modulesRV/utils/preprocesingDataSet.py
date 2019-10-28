########################################################################
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

import pandas as pd
from sklearn import preprocessing

class preprocessingUtil(object):

    def __init__(self, dataSet, optionScale):

        self.dataSet = dataSet
        self.optionScale = optionScale

    def applyScale(self):

        if self.optionScale == 1:#quick scale
            self.dataSet = preprocessing.scale(self.dataSet)

        elif self.optionScale == 2:#standar scale
            scaler = preprocessing.StandardScaler().fit(self.dataSet)
            self.dataSet = scaler.transform(self.dataSet)

        elif self.optionScale == 3:#min max scaler
            min_max_scaler = preprocessing.MinMaxScaler()
            self.dataSet = min_max_scaler.fit_transform(self.dataSet)

        elif self.optionScale == 4:#quantile transformation
            quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
            self.dataSet = quantile_transformer.fit_transform(self.dataSet)

        else:#powerTransformation
            pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
            self.dataSet = pt.fit_transform(self.dataSet)
