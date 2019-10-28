########################################################################
# createHistogram.py,
#
# Make a histogram of a dataset.
# Receives a dataset,a column key of it, and the filename ofr the output
#
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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns
from pandas.tools.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler
import pandas.tools.plotting as pdplt

class histograme(object):

    def __init__(self, dataSet):

        self.dataSet = dataSet

    #funcion que permite crear un histograma...
    def generateHistogram(self, key, exportName, title):

        #plt.figure()
        sns.set(color_codes=True)
        sns.set(style="ticks")
        sns_plot = sns.distplot(self.dataSet[key] , color="olive", label=key, kde=False, rug=True)
        sns.plt.legend()
        sns.plt.title(title)
        sns_plot.figure.savefig(exportName)
